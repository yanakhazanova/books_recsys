from pathlib import Path

# Install log capture BEFORE other imports so their prints are recorded too.
from app.services.log_capture import install_log_capture
LOG_FILE = install_log_capture()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.api.endpoints import router
from app.services.train_model import AFMTrainer, DataCache
from app.core.config import settings
from app.state import (
    model, candidates, train_ratings, test_ratings,
    book_features, user_features, triplets, books_df, books_titles_dict, data_cache
)
import app.state as state

app = FastAPI(
    title="Book Recommendation System",
    description="API для рекомендаций книг с SHAP интерпретацией",
    version="1.0.0"
)

# CORS для тестирования из браузера
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Отключаем браузерное кэширование статики, чтобы изменения в app.js
# применялись немедленно без ручного hard-refresh.
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

class NoCacheStaticMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"
        return response

app.add_middleware(NoCacheStaticMiddleware)

# Serve frontend
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.on_event("startup")
async def startup_event():
    """При старте сервера загружаем модель и данные из кэша"""
    from app.services.train_model import AFMTrainer
    
    print("\n" + "="*60)
    print("🚀 ЗАПУСК FASTAPI ПРИЛОЖЕНИЯ")
    print("="*60)
    
    # 1. Загружаем очищенные данные для получения books_df
    ratings, books, users = data_cache.load_cleaned_data()
    if books is not None:
        state.books_df = books
        if 'ISBN' in books.columns and 'Title' in books.columns:
            state.books_titles_dict = dict(zip(
                books['ISBN'].astype(str), 
                books['Title']
            ))
            print(f"✅ Загружены данные о книгах: {len(state.books_df)} записей, {len(state.books_titles_dict)} названий")
    
    # 2. Загружаем сплиты
    train_r, test_r = data_cache.load_split()
    if train_r is not None:
        state.train_ratings = train_r
        state.test_ratings = test_r
        print(f"✅ Загружены сплиты: train={len(state.train_ratings)}, test={len(state.test_ratings)}")
    
    # 3. Загружаем кандидатов
    state.candidates = data_cache.load_candidates()
    if state.candidates is not None:
        print(f"✅ Загружены кандидаты для {len(state.candidates)} пользователей")
    
    # 4. Загружаем фичи и триплеты
    bf, uf, trips = data_cache.load_features()
    if bf is not None:
        state.book_features = bf
        state.user_features = uf
        state.triplets = trips
        print(f"✅ Загружены фичи: books={state.book_features.shape}, users={state.user_features.shape}")
    
    # 5. Загружаем модель
    if settings.model_path.exists():
        try:
            trainer = AFMTrainer()
            trainer.load(str(settings.model_path))
            trainer.user_features = state.user_features
            trainer.book_features = state.book_features
            state.model = trainer
            print("✅ Модель успешно загружена")
        except Exception as e:
            print(f"⚠️ Не удалось загрузить модель: {e}")
    
    print("="*60 + "\n")

@app.get("/")
async def root():
    """Раздаёт frontend SPA, либо JSON если фронт не собран."""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {
        "message": "Book Recommendation System API",
        "endpoints": [
            "POST /api/v1/prepare - подготовить данные",
            "POST /api/v1/train - обучить модель",
            "GET /api/v1/metrics?k=10 - получить метрики",
            "GET /api/v1/shap/global - глобальная SHAP интерпретация",
            "GET /api/v1/recommend/{user_id}?n_recs=10 - рекомендации для пользователя"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
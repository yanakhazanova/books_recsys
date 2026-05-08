from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router
from app.services.train_model import AFMTrainer, DataCache
from app.core.config import settings

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

@app.get("/")
async def root():
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


@app.on_event("startup")
async def startup_event():
    """При старте сервера загружаем модель и данные из кэша"""
    global model, candidates, train_ratings, test_ratings, book_features, user_features, triplets, books_df
    
    print("\n" + "="*60)
    print("🚀 ЗАПУСК FASTAPI ПРИЛОЖЕНИЯ")
    print("="*60)
    
    # 1. Загружаем очищенные данные для получения books_df
    cache = DataCache()
    ratings, books, users = cache.load_cleaned_data()
    if books is not None:
        books_df = books
        print(f"✅ Загружены данные о книгах: {len(books_df)} записей")
    
    # 2. Загружаем сплиты
    train_ratings, test_ratings = cache.load_split()
    if train_ratings is not None:
        print(f"✅ Загружены сплиты: train={len(train_ratings)}, test={len(test_ratings)}")
    
    # 3. Загружаем кандидатов
    candidates = cache.load_candidates()
    if candidates is not None:
        print(f"✅ Загружены кандидаты для {len(candidates)} пользователей")
    
    # 4. Загружаем фичи и триплеты
    book_features, user_features, triplets = cache.load_features()
    if book_features is not None:
        print(f"✅ Загружены фичи: books={book_features.shape}, users={user_features.shape}, triplets={len(triplets)}")
    
    # 5. Загружаем модель
    model_path = settings.model_path
    if model_path.exists():
        try:
            print(f"📂 Загружаем модель из {model_path}")
            trainer = AFMTrainer()
            trainer.load(str(model_path))
            
            # Восстанавливаем ссылки на данные для предсказаний
            trainer.user_features = user_features
            trainer.book_features = book_features
            trainer.user_clean_to_original = getattr(trainer, 'user_clean_to_original', {})
            trainer.book_clean_to_original = getattr(trainer, 'book_clean_to_original', {})
            
            model = trainer
            print("✅ Модель успешно загружена в память")
        except Exception as e:
            print(f"⚠️ Не удалось загрузить модель: {e}")
    else:
        print("ℹ️ Сохраненная модель не найдена")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
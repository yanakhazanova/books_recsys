from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router
from app.services.train_model import AFMTrainer
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
    """При старте сервера пытаемся загрузить существующую модель"""
    global model
    
    model_path = settings.model_path
    if model_path.exists():
        try:
            print(f"📂 Загружаем существующую модель из {model_path}")
            trainer = AFMTrainer()
            model = trainer.load(str(model_path))
            print("✅ Модель успешно загружена")
        except Exception as e:
            print(f"⚠️ Не удалось загрузить модель: {e}")
    else:
        print("ℹ️ Сохраненная модель не найдена. Для использования выполните POST /train")
        

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
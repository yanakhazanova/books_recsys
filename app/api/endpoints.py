from pathlib import Path
import numpy as np

from app.services import (
    data_pipeline,
    collab_filter,
    feature_eng,
    pairwise_data,
    train_model,
    metrics,
    shap_analysis,
    user_recs,
    metrics_calculation
)

# app/api/endpoints.py
from fastapi import APIRouter, HTTPException, Query
from app.core.schemas import (
    PrepareDataResponse, TrainResponse, MetricsResponse,
    GlobalShapResponse, UserRecsResponse, CandidatesResponse,
)
from app.services.data_pipeline import DataLoader, DataAnalyzer
from app.services.train_model import (
    TrainTestSplitter, CollaborativeGenerator, 
    FeaturePipeline, AFMTrainer, DataCache
)
from app.services.metrics_calculation import MetricsCalculator
from app.core.config import settings
from typing import Dict, List, Tuple


router = APIRouter(prefix="/api/v1", tags=["recommendations"])

# Глобальные переменные для хранения состояния
train_ratings = None
test_ratings = None
candidates = None
book_features = None
user_features = None
triplets = None
model = None
books_df = None


# Кэш для хранения промежуточных данных
data_cache = DataCache()


@router.post("/prepare", response_model=PrepareDataResponse)
async def prepare_data():
    """Подготовка данных: загрузка, очистка, сохранение"""
    global books_df

    try:
        loader = DataLoader()
        result = loader.run_preparation()
        
        # Анализируем данные
        ratings = loader.ratings
        analyzer = DataAnalyzer()
        metrics = analyzer.analyze_ratings(ratings)
        loader.load_data()
        books_df = loader.books
        
        # Сохраняем очищенные данные в кэш
        data_cache.save_cleaned_data(loader.ratings, loader.books, loader.users)
        
        return PrepareDataResponse(
            status="success",
            n_users=result['n_users'],
            n_books=result['n_books'],
            n_interactions=result['n_interactions'],
            message=result['message']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/split", response_model=TrainResponse)
async def split_train_test(
    test_items_per_user: int = Query(2, description="Сколько книг на пользователя в тесте")
):
    """Только разделение данных на train/test"""
    global train_ratings, test_ratings
    
    try:
        # Загружаем очищенные данные из кэша
        ratings, books, users = data_cache.load_cleaned_data()
        
        if ratings is None:
            raise HTTPException(status_code=400, detail="Данные не подготовлены. Сначала вызовите /prepare")
        
        # Разделяем на train/test
        splitter = TrainTestSplitter(
            strong_pos_threshold=8,
            weak_pos_is_zero=True,
            neg_threshold=5,
            test_items_per_user=test_items_per_user,
            min_strong_pos=3
        )
        train_ratings, test_ratings = splitter.split_all_users(ratings)
        
        # Сохраняем сплиты в кэш
        data_cache.save_split(train_ratings, test_ratings)
        
        return TrainResponse(
            status="success",
            model_path="data/processed/split_completed",
            cv_score=None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/candidates", response_model=CandidatesResponse)
async def generate_candidates(
    n_candidates: int = Query(1000, description="Количество кандидатов на пользователя")
):
    """Генерация кандидатов коллаборативной фильтрацией"""
    global train_ratings, test_ratings, candidates
    
    try:
        # Загружаем сплиты из кэша
        if train_ratings is None:
            train_ratings, test_ratings = data_cache.load_split()
        
        if train_ratings is None:
            raise HTTPException(status_code=400, detail="Данные не разделены. Сначала вызовите /split")
        
        # Генерируем кандидатов
        generator = CollaborativeGenerator(n_candidates=n_candidates)
        candidates = generator.generate(train_ratings, test_ratings)
        
        # Сохраняем кандидатов в кэш
        data_cache.save_candidates(candidates)
        
        # Статистика
        n_users = len(candidates)
        avg_candidates = sum(len(recs) for recs in candidates.values()) / n_users
        
        return CandidatesResponse(
            status="success",
            n_users_with_candidates=n_users,
            avg_candidates_per_user=avg_candidates,
            candidates_path="models/collaborative_candidates.pkl"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/features", response_model=TrainResponse)
async def create_features():
    """Формирование фичей и триплетов"""
    global train_ratings, candidates, book_features, user_features, triplets
    
    try:
        # Загружаем необходимые данные
        if train_ratings is None:
            train_ratings, test_ratings = data_cache.load_split()
        
        if candidates is None:
            candidates = data_cache.load_candidates()
        
        # Загружаем книги и пользователей
        ratings, books, users = data_cache.load_cleaned_data()
        
        if train_ratings is None or candidates is None:
            raise HTTPException(status_code=400, detail="Не хватает данных. Сначала вызовите /split и /candidates")
        
        # Формируем фичи и триплеты
        feature_pipeline = FeaturePipeline()
        book_features, user_features, triplets = feature_pipeline.generate_features_and_triplets(
            train_ratings=train_ratings,
            books=books,
            users=users
        )
        
        # Сохраняем
        feature_pipeline.save()
        data_cache.save_features(book_features, user_features, triplets)
        
        return TrainResponse(
            status="success",
            model_path="models/features_and_scalers.pkl",
            cv_score=None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@router.get("/training/losses")
async def get_training_losses():
    """Получить историю потерь при обучении"""
    global model
    
    if model is None:
        raise HTTPException(status_code=400, detail="Модель еще не обучена")
    
    return {
        "train_losses": model.train_losses,
        "val_losses": model.val_losses
    }


# app/api/endpoints.py - исправленные эндпоинты

# app/api/endpoints.py - исправленный эндпоинт train

@router.post("/train", response_model=TrainResponse)
async def train_model(
    use_existing_model: bool = Query(False, description="Использовать уже сохраненную модель"),
    epochs: int = Query(10, ge=1, le=100),
    batch_size: int = Query(512, ge=1, le=10000),
    embed_dim: int = Query(64, ge=16, le=256),
    attention_dim: int = Query(32, ge=8, le=128),
    learning_rate: float = Query(0.001, ge=0.0001, le=0.1)
):
    """Обучение модели AFM (или загрузка существующей)"""
    global model, book_features, user_features, triplets, candidates, train_ratings, test_ratings
    
    try:
        # Загружаем все необходимые данные из кэша
        if book_features is None:
            book_features, user_features, triplets = data_cache.load_features()
        
        if candidates is None:
            candidates = data_cache.load_candidates()
        
        if train_ratings is None:
            train_ratings, test_ratings = data_cache.load_split()
        
        if book_features is None or triplets is None:
            raise HTTPException(status_code=400, detail="Нет фичей. Сначала вызовите /features")
        
        if candidates is None:
            raise HTTPException(status_code=400, detail="Нет кандидатов. Сначала вызовите /candidates")
        
        trainer = AFMTrainer(
            embed_dim=embed_dim,
            attention_dim=attention_dim,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate
        )
        
        model_path = Path(settings.models_dir) / "afm_model.pth"
        
        if use_existing_model and model_path.exists():
            print(f"📂 Загружаем существующую модель из {model_path}")
            trainer.load(str(model_path))
            
            # ВАЖНО: восстанавливаем ссылки на данные для предсказаний
            trainer.user_features = user_features
            trainer.book_features = book_features
            
            model = trainer
            print("✅ Модель загружена и готова к использованию")
        else:
            if use_existing_model and not model_path.exists():
                print(f"⚠️ Модель не найдена в {model_path}, обучаем новую...")
            
            print("🚀 Обучаем новую модель...")
            trainer.fit(triplets, user_features, book_features)
            trainer.save(str(model_path))
            
            # Сохраняем ссылки на данные
            trainer.user_features = user_features
            trainer.book_features = book_features
            
            model = trainer
        
        return TrainResponse(
            status="success",
            model_path=str(model_path),
            cv_score=model.val_losses[-1] if model.val_losses else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Ошибка: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")

@router.post("/train_full", response_model=TrainResponse)
async def train_full_pipeline(
    test_items_per_user: int = Query(2, ge=1, le=10, description="Сколько книг на пользователя в тесте"),
    n_candidates: int = Query(1000, ge=100, le=5000, description="Количество кандидатов на пользователя"),
    use_existing_model: bool = Query(False, description="Использовать уже сохраненную модель"),
    epochs: int = Query(10, ge=1, le=100, description="Количество эпох обучения"),
    batch_size: int = Query(512, ge=1, le=10000, description="Размер батча"),
    embed_dim: int = Query(64, ge=16, le=256, description="Размерность эмбеддингов"),
    attention_dim: int = Query(32, ge=8, le=128, description="Размерность attention слоя"),
    learning_rate: float = Query(0.001, ge=0.0001, le=0.1, description="Скорость обучения"),
    skip_if_exists: bool = Query(False, description="Пропускать этапы, если данные уже готовы")
):
    """
    Полный пайплайн обучения одной командой
    
    - skip_if_exists=True: пропускает готовые этапы (продолжает с того места, где остановились)
    - skip_if_exists=False: перезапускает все этапы заново
    """
    try:
        results = {}
        
        # Этап 1: Подготовка данных
        if not skip_if_exists or not data_cache.check_cleaned_data():
            results['prepare'] = await prepare_data()
        else:
            results['prepare'] = {"status": "skipped", "message": "Данные уже подготовлены"}
        
        # Этап 2: Разделение
        if not skip_if_exists or not data_cache.check_split():
            results['split'] = await split_train_test(test_items_per_user=test_items_per_user)
        else:
            results['split'] = {"status": "skipped", "message": "Сплит уже выполнен"}
        
        # Этап 3: Генерация кандидатов
        if not skip_if_exists or not data_cache.check_candidates():
            results['candidates'] = await generate_candidates(n_candidates=n_candidates)
        else:
            results['candidates'] = {"status": "skipped", "message": "Кандидаты уже сгенерированы"}
        
        # Этап 4: Формирование фичей
        if not skip_if_exists or not data_cache.check_features():
            results['features'] = await create_features()
        else:
            results['features'] = {"status": "skipped", "message": "Фичи уже сформированы"}
        
        # Этап 5: Обучение модели (с передачей всех параметров)
        if not skip_if_exists or not data_cache.check_model():
            results['train'] = await train_model(
                use_existing_model=use_existing_model,
                epochs=epochs,
                batch_size=batch_size,
                embed_dim=embed_dim,
                attention_dim=attention_dim,
                learning_rate=learning_rate
            )
        else:
            results['train'] = {"status": "skipped", "message": "Модель уже обучена"}
        
        return TrainResponse(
            status="success",
            model_path=str(settings.model_path),
            cv_score=results.get('train', {}).get('cv_score', None)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Ошибка в полном пайплайне: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ошибка в полном пайплайне: {str(e)}")
    

# app/api/endpoints.py - исправленный эндпоинт status

def convert_to_serializable(obj):
    """Рекурсивно конвертирует numpy типы в Python типы"""
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj


@router.get("/status", response_model=dict)
async def get_training_status():
    """Проверка статуса подготовки данных и обучения"""
    global train_ratings, test_ratings, candidates, book_features, user_features, triplets, model, books_df
    
    # Проверяем наличие сохраненной модели AFM
    model_file_exists = (settings.model_path).exists()
    
    # Проверяем, загружена ли модель в память
    model_in_memory = model is not None
    
    # Получаем информацию о модели, если она загружена
    model_info = {}
    if model_in_memory and hasattr(model, 'model'):
        model_info = {
            "embed_dim": getattr(model, 'embed_dim', None),
            "attention_dim": getattr(model, 'attention_dim', None),
            "is_trained": len(getattr(model, 'train_losses', [])) > 0,
            "epochs_completed": len(getattr(model, 'train_losses', [])),
            "final_train_loss": float(model.train_losses[-1]) if model.train_losses else None,
            "final_val_loss": float(model.val_losses[-1]) if model.val_losses else None
        }
    
    # Получаем статистику по фичам
    features_info = {}
    if book_features is not None:
        features_info = {
            "book_features_shape": list(book_features.shape),
            "user_features_shape": list(user_features.shape) if user_features is not None else None,
            "triplets_count": int(len(triplets)) if triplets is not None else 0
        }
    
    # Статистика по кандидатам
    candidates_info = {}
    if candidates is not None:
        candidates_counts = [len(recs) for recs in candidates.values()]
        candidates_info = {
            "total_users": len(candidates),
            "avg_candidates": float(np.mean(candidates_counts)) if candidates_counts else 0,
            "min_candidates": int(np.min(candidates_counts)) if candidates_counts else 0,
            "max_candidates": int(np.max(candidates_counts)) if candidates_counts else 0
        }
    
    # Получаем количество пользователей
    train_users_count = 0
    test_users_count = 0
    train_interactions = 0
    test_interactions = 0
    
    if train_ratings is not None:
        train_users_count = int(train_ratings['User-ID'].nunique())
        train_interactions = len(train_ratings)
    
    if test_ratings is not None:
        test_users_count = int(test_ratings['User-ID'].nunique())
        test_interactions = len(test_ratings)
    
    status = {
        "data_ready": data_cache.check_cleaned_data(),
        "split_ready": data_cache.check_split(),
        "candidates_ready": data_cache.check_candidates(),
        "features_ready": data_cache.check_features(),
        "model_file_exists": model_file_exists,
        "model_loaded_in_memory": model_in_memory,
        "train_users": train_users_count,
        "test_users": test_users_count,
        "total_interactions_train": train_interactions,
        "total_interactions_test": test_interactions,
        "candidates": candidates_info,
        "features": features_info,
        "model": model_info,
        "next_steps": _get_next_steps(
            data_cache.check_cleaned_data(),
            data_cache.check_split(),
            data_cache.check_candidates(),
            data_cache.check_features(),
            model_file_exists,
            model_in_memory
        )
    }
    
    # Конвертируем все numpy типы в стандартные Python типы
    status = convert_to_serializable(status)
    
    return status

def _get_next_steps(data_ready, split_ready, candidates_ready, features_ready, model_file_exists, model_in_memory):
    """Определяет, какой эндпоинт вызывать следующим"""
    if not data_ready:
        return "Вызовите POST /api/v1/prepare для загрузки и очистки данных"
    
    if not split_ready:
        return "Вызовите POST /api/v1/split для разделения данных на train/test"
    
    if not candidates_ready:
        return "Вызовите POST /api/v1/candidates для генерации кандидатов"
    
    if not features_ready:
        return "Вызовите POST /api/v1/features для формирования фичей и триплетов"
    
    if not model_in_memory:
        if model_file_exists:
            return "Модель существует на диске. Вызовите POST /api/v1/train?use_existing_model=true для загрузки"
        else:
            return "Вызовите POST /api/v1/train?use_existing_model=false для обучения новой модели"
    
    return "Все готово! Используйте GET /api/v1/metrics и GET /api/v1/recommend/{user_id}"

@router.delete("/cache")
async def clear_cache():
    """Очищает весь кэш промежуточных данных"""
    global train_ratings, test_ratings, candidates, book_features, user_features, triplets, model
    
    data_cache.clear()
    
    # Очищаем глобальные переменные
    train_ratings = None
    test_ratings = None
    candidates = None
    book_features = None
    user_features = None
    triplets = None
    model = None
    
    return {"status": "success", "message": "Кэш очищен"}


# app/api/endpoints.py - добавить/обновить эндпоинты

@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(k: int = Query(10, ge=1, le=50)):
    """Расчет метрик на тестовых пользователях"""
    global model, candidates, test_ratings, book_features, user_features
    
    # Проверяем наличие всех необходимых данных
    if model is None:
        raise HTTPException(status_code=400, detail="Модель не загружена. Сначала вызовите /train")
    
    if candidates is None:
        candidates = data_cache.load_candidates()
        if candidates is None:
            raise HTTPException(status_code=400, detail="Нет кандидатов. Сначала вызовите /candidates")
    
    if test_ratings is None:
        _, test_ratings = data_cache.load_split()
        if test_ratings is None:
            raise HTTPException(status_code=400, detail="Нет тестовых данных. Сначала вызовите /split")
    
    # Убеждаемся, что у модели есть ссылки на данные
    if model.user_features is None:
        model.user_features = user_features
    if model.book_features is None:
        model.book_features = book_features
    
    try:
        print(f"\n🎯 Генерация рекомендаций для расчета метрик @{k}...")
        recommendations = model.predict_for_all_users(candidates, top_n=k)
        
        # Рассчитываем метрики
        metrics = MetricsCalculator.calculate_all_metrics(recommendations, test_ratings, k)
        
        return MetricsResponse(
            ndcg_at_k=float(metrics['ndcg_at_k']),
            precision_at_k=float(metrics['precision_at_k']),
            recall_at_k=float(metrics['recall_at_k']),
            k=k,
            hit_rate_at_k=float(metrics.get('hit_rate', 0)),
            users_with_hits_ratio=float(metrics.get('users_with_hits_ratio', 0))
        )
        
    except Exception as e:
        print(f"❌ Ошибка при расчете метрик: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")

@router.get("/recommend/{user_id}", response_model=UserRecsResponse)
async def recommend_for_user(
    user_id: int,
    n_recs: int = Query(10, ge=1, le=50, description="Количество рекомендаций")
):
    """Рекомендации для конкретного пользователя"""
    global model, candidates, book_features, user_features, books_df
    
    if model is None:
        raise HTTPException(status_code=400, detail="Модель еще не обучена. Сначала вызовите /train")
    
    if candidates is None:
        raise HTTPException(status_code=400, detail="Нет кандидатов. Сначала вызовите /candidates")
    
    try:
        # Проверяем, есть ли пользователь в кандидатах
        if user_id not in candidates:
            raise HTTPException(status_code=404, detail=f"Пользователь {user_id} не найден в кандидатах")
        
        # Получаем рекомендации
        candidate_books = list(candidates[user_id].keys())
        ranked = model.predict_for_user(user_id, candidate_books, top_n=n_recs)
        
        # Формируем ответ
        recommendations = []
        for i, (book_id, score) in enumerate(ranked):
            # Пытаемся получить название книги
            book_title = None
            if books_df is not None:
                book_row = books_df[books_df['ISBN'] == book_id]
                if not book_row.empty:
                    book_title = book_row.iloc[0].get('Title', None)
            
            recommendations.append({
                "book_id": book_id,
                "book_title": book_title,
                "score": score,
                "rank": i + 1
            })
        
        # TODO: Добавить SHAP интерпретацию
        shap_dict = {}
        explanation_text = "SHAP интерпретация будет добавлена позже"
        
        return UserRecsResponse(
            user_id=user_id,
            recommendations=recommendations,
            shap_interpretation=shap_dict,
            explanation_text=explanation_text
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Ошибка при получении рекомендаций: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ошибка при получении рекомендаций: {str(e)}")


@router.post("/recommend/batch")
async def recommend_batch(
    user_ids: List[int],
    n_recs: int = Query(10, description="Количество рекомендаций на пользователя")
):
    """Рекомендации для нескольких пользователей (batch режим)"""
    global model, candidates
    
    if model is None:
        raise HTTPException(status_code=400, detail="Модель еще не обучена")
    
    results = {}
    for user_id in user_ids:
        if user_id in candidates:
            candidate_books = list(candidates[user_id].keys())
            ranked = model.predict_for_user(user_id, candidate_books, top_n=n_recs)
            results[user_id] = [{"book_id": book_id, "score": score} for book_id, score in ranked]
        else:
            results[user_id] = []
    
    return {"recommendations": results}


@router.get("/shap/global", response_model=GlobalShapResponse)
async def get_global_shap():
    """Глобальная SHAP интерпретация"""
    global model
    
    if model is None:
        raise HTTPException(status_code=400, detail="Модель еще не обучена. Сначала вызовите /train")
    
    # TODO: Реализовать SHAP анализ
    return GlobalShapResponse(
        feature_importance={
            "collab_score": 0.45,
            "book_popularity": 0.32,
            "user_avg_rating": 0.18,
            "genre_match": 0.05
        },
        shap_values_shape=[100, 4],
        message="Глобальная SHAP интерпретация (заглушка)"
    )

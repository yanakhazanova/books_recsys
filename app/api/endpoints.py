# app/api/endpoints.py
from fastapi import APIRouter, HTTPException, Query
from app.core.schemas import (
    PrepareDataResponse, TrainResponse, MetricsResponse,
    GlobalShapResponse, UserRecsResponse
)
from app.services.data_pipeline import DataLoader, DataAnalyzer
from app.services.train_model import (
    TrainTestSplitter, CollaborativeGenerator, 
    FeaturePipeline, AFMTrainer
)
from app.services.metrics_calculation import MetricsCalculator
from app.core.config import settings
from app.state import data_cache
import app.state as state

router = APIRouter(prefix="/api/v1", tags=["recommendations"])


@router.post("/prepare", response_model=PrepareDataResponse)
async def prepare_data():
    """Подготовка данных: загрузка, очистка, сохранение"""
    try:
        loader = DataLoader()
        result = loader.run_preparation()
        
        # Обновляем состояние
        state.books_df = loader.books
        if 'ISBN' in state.books_df.columns and 'Title' in state.books_df.columns:
            state.books_titles_dict = dict(zip(
                state.books_df['ISBN'].astype(str), 
                state.books_df['Title']
            ))
            print(f"✅ Загружено названий для {len(state.books_titles_dict)} книг")
        
        ratings = loader.ratings
        analyzer = DataAnalyzer()
        metrics = analyzer.analyze_ratings(ratings)
        
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
        state.train_ratings, state.test_ratings = splitter.split_all_users(ratings)
        
        # Сохраняем сплиты в кэш
        data_cache.save_split(state.train_ratings, state.test_ratings)
        
        return TrainResponse(
            status="success",
            model_path="data/processed/split_completed",
            cv_score=None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/candidates", response_model=TrainResponse)
async def generate_candidates(
    n_candidates: int = Query(1000, description="Количество кандидатов на пользователя")
):
    """Генерация кандидатов коллаборативной фильтрацией"""
    global train_ratings, test_ratings, candidates
    
    try:
        # Загружаем сплиты из кэша
        if state.train_ratings is None:
            state.train_ratings, state.test_ratings = data_cache.load_split()
        
        if state.train_ratings is None:
            raise HTTPException(status_code=400, detail="Данные не разделены. Сначала вызовите /split")
        
        # Генерируем кандидатов
        generator = CollaborativeGenerator(n_candidates=n_candidates)
        state.candidates = generator.generate(state.train_ratings, state.test_ratings)
        
        # Сохраняем кандидатов в кэш
        data_cache.save_candidates(state.candidates)
        
        return TrainResponse(
            status="success",
            model_path="models/collaborative_candidates.pkl",
            cv_score=None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/features", response_model=TrainResponse)
async def create_features():
    """Формирование фичей и триплетов"""
    global train_ratings, candidates, book_features, user_features, triplets
    
    try:
        # Загружаем необходимые данные
        if state.train_ratings is None:
            state.train_ratings, state.test_ratings = data_cache.load_split()
        
        if state.candidates is None:
            state.candidates = data_cache.load_candidates()
        
        # Загружаем книги и пользователей
        ratings, books, users = data_cache.load_cleaned_data()
        
        if state.train_ratings is None or state.candidates is None:
            raise HTTPException(status_code=400, detail="Не хватает данных. Сначала вызовите /split и /candidates")
        
        # Формируем фичи и триплеты
        feature_pipeline = FeaturePipeline()
        state.book_features, state.user_features, state.triplets = feature_pipeline.generate_features_and_triplets(
            train_ratings=state.train_ratings,
            books=books,
            users=users
        )
        
        # Сохраняем
        feature_pipeline.save()
        data_cache.save_features(state.book_features, state.user_features, state.triplets)
        
        return TrainResponse(
            status="success",
            model_path="models/features_and_scalers.pkl",
            cv_score=None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    try:
        # Загружаем фичи и триплеты
        if state.book_features is None:
            state.book_features, state.user_features, state.triplets = data_cache.load_features()
        
        if state.book_features is None or state.triplets is None:
            raise HTTPException(status_code=400, detail="Нет фичей. Сначала вызовите /features")
        
        trainer = AFMTrainer(
            embed_dim=embed_dim,
            attention_dim=attention_dim,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate
        )
        
        model_path = settings.model_path
        
        if use_existing_model and model_path.exists():
            print(f"📂 Загружаем существующую модель из {model_path}")
            trainer.load(str(model_path))
        else:
            if use_existing_model and not model_path.exists():
                print(f"⚠️ Модель не найдена в {model_path}, обучаем новую...")
            
            print("🚀 Обучаем новую модель...")
            trainer.fit(state.triplets, state.user_features, state.book_features)
            trainer.save(str(model_path))
        
        # Сохраняем ссылки на данные
        trainer.user_features = state.user_features
        trainer.book_features = state.book_features
        state.model = trainer
        
        return TrainResponse(
            status="success",
            model_path=str(model_path),
            cv_score=trainer.val_losses[-1] if trainer.val_losses else None
        )
        
    except Exception as e:
        print(f"❌ Ошибка: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train_full", response_model=TrainResponse)
async def train_full_pipeline(
    test_items_per_user: int = Query(2, description="Сколько книг на пользователя в тесте"),
    n_candidates: int = Query(1000, description="Количество кандидатов на пользователя"),
    use_existing_model: bool = Query(False, description="Использовать уже сохраненную модель"),
    epochs: int = Query(10, ge=1, le=100, description="Количество эпох обучения"),
    batch_size: int = Query(512, ge=1, le=10000, description="Размер батча"),
    embed_dim: int = Query(64, ge=16, le=256, description="Размерность эмбеддингов"),
    attention_dim: int = Query(32, ge=8, le=128, description="Размерность attention слоя"),
    learning_rate: float = Query(0.001, ge=0.0001, le=0.1, description="Скорость обучения"),
    skip_if_exists: bool = Query(False, description="Пропускать готовые этапы")
):
    """
    Полный пайплайн обучения одной командой
    """
    try:
        # Этап 1: Подготовка данных
        if not skip_if_exists or not data_cache.check_cleaned_data():
            await prepare_data()
        
        # Этап 2: Разделение
        if not skip_if_exists or not data_cache.check_split():
            await split_train_test(test_items_per_user=test_items_per_user)
        
        # Этап 3: Генерация кандидатов
        if not skip_if_exists or not data_cache.check_candidates():
            await generate_candidates(n_candidates=n_candidates)
        
        # Этап 4: Формирование фичей
        if not skip_if_exists or not data_cache.check_features():
            await create_features()
        
        # Этап 5: Обучение модели
        result = await train_model(
            use_existing_model=use_existing_model,
            epochs=epochs,
            batch_size=batch_size,
            embed_dim=embed_dim,
            attention_dim=attention_dim,
            learning_rate=learning_rate
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Ошибка: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommend/{user_id}", response_model=UserRecsResponse)
async def recommend_for_user(
    user_id: int,
    n_recs: int = Query(10, ge=1, le=50)
):
    """Рекомендации для конкретного пользователя"""
    if state.model is None:
        raise HTTPException(status_code=400, detail="Модель не загружена. Сначала вызовите /train")
    
    if state.candidates is None:
        state.candidates = data_cache.load_candidates()
        if state.candidates is None:
            raise HTTPException(status_code=400, detail="Нет кандидатов. Сначала вызовите /candidates")
    
    try:
        if user_id not in state.candidates:
            raise HTTPException(status_code=404, detail=f"Пользователь {user_id} не найден")
        
        candidate_books = list(state.candidates[user_id].keys())
        ranked = state.model.predict_for_user(user_id, candidate_books, top_n=n_recs)
        
        recommendations = []
        for i, (book_id, score) in enumerate(ranked):
            book_title = None
            if state.books_titles_dict is not None:
                book_title = state.books_titles_dict.get(str(book_id))
            
            recommendations.append({
                "book_id": book_id,
                "book_title": book_title,
                "score": score,
                "rank": i + 1
            })
        
        return UserRecsResponse(
            user_id=user_id,
            recommendations=recommendations,
            shap_interpretation={},
            explanation_text="SHAP интерпретация будет добавлена позже"
        )
        
    except Exception as e:
        print(f"❌ Ошибка: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    k: int = Query(10, ge=1, le=50),
    sample_users: int = Query(100, ge=1, le=10000)
):
    """Расчет метрик на тестовых пользователях"""
    if state.model is None:
        raise HTTPException(status_code=400, detail="Модель не загружена. Сначала вызовите /train")
    
    if state.candidates is None:
        state.candidates = data_cache.load_candidates()
        if state.candidates is None:
            raise HTTPException(status_code=400, detail="Нет кандидатов. Сначала вызовите /candidates")
    
    if state.test_ratings is None:
        _, state.test_ratings = data_cache.load_split()
        if state.test_ratings is None:
            raise HTTPException(status_code=400, detail="Нет тестовых данных. Сначала вызовите /split")
    
    try:
        # Получаем тестовых пользователей
        test_users_list = list(state.test_ratings['User-ID'].unique())
        
        # Ограничиваем выборку
        if sample_users < len(test_users_list):
            import random
            random.seed(42)
            sampled_users = random.sample(test_users_list, sample_users)
        else:
            sampled_users = test_users_list
        
        # Фильтруем кандидатов
        filtered_candidates = {u: state.candidates[u] for u in sampled_users if u in state.candidates}
        
        # Генерируем рекомендации
        recommendations = state.model.predict_for_all_users(filtered_candidates, top_n=k)
        
        # Фильтруем тестовые данные
        filtered_test = state.test_ratings[state.test_ratings['User-ID'].isin(sampled_users)]
        
        # Рассчитываем метрики
        metrics = MetricsCalculator.calculate_all_metrics(recommendations, filtered_test, k)
        
        return MetricsResponse(
            ndcg_at_k=float(metrics['ndcg_at_k']),
            precision_at_k=float(metrics['precision_at_k']),
            recall_at_k=float(metrics['recall_at_k']),
            k=k,
            hit_rate_at_k=float(metrics.get('hit_rate', 0)),
            users_with_hits_ratio=float(metrics.get('users_with_hits_ratio', 0)),
            sample_size=len(sampled_users),
            total_users=len(test_users_list)
        )
        
    except Exception as e:
        print(f"❌ Ошибка: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/status", response_model=dict)
async def get_training_status():
    """Проверка статуса подготовки данных и обучения"""
    
    # Получаем статусы
    data_ready = data_cache.check_cleaned_data()
    split_ready = data_cache.check_split()
    candidates_ready = data_cache.check_candidates()
    features_ready = data_cache.check_features()
    model_file_exists = settings.model_path.exists()
    model_loaded = state.model is not None
    
    # Формируем базовый ответ
    status = {
        "data_ready": data_ready,
        "split_ready": split_ready,
        "candidates_ready": candidates_ready,
        "features_ready": features_ready,
        "model_file_exists": model_file_exists,
        "model_loaded_in_memory": model_loaded,
        "candidates_count": len(state.candidates) if state.candidates else 0,
        "train_users": state.train_ratings['User-ID'].nunique() if state.train_ratings is not None else 0,
        "test_users": state.test_ratings['User-ID'].nunique() if state.test_ratings is not None else 0,
        "books_titles_count": len(state.books_titles_dict) if state.books_titles_dict else 0,
        "next_steps": _get_next_steps(
            data_ready, split_ready, candidates_ready, 
            features_ready, model_file_exists, model_loaded
        )
    }
    
    return status


def _get_next_steps(data_ready, split_ready, candidates_ready, features_ready, model_file_exists, model_loaded):
    """Определяет, какой эндпоинт вызывать следующим"""
    
    if not data_ready:
        return "📋 Вызовите POST /api/v1/prepare для загрузки и очистки данных"
    
    if not split_ready:
        return "✂️ Вызовите POST /api/v1/split для разделения данных на train/test"
    
    if not candidates_ready:
        return "🎯 Вызовите POST /api/v1/candidates для генерации кандидатов коллаборативной фильтрацией"
    
    if not features_ready:
        return "🔧 Вызовите POST /api/v1/features для формирования фичей и триплетов"
    
    if not model_loaded:
        if model_file_exists:
            return "🤖 Модель существует на диске. Вызовите POST /api/v1/train?use_existing_model=true для загрузки"
        else:
            return "🚀 Вызовите POST /api/v1/train?use_existing_model=false для обучения новой модели"
    
    # Если всё готово
    recommendations_count = state.candidates_count if hasattr(state, 'candidates_count') else 0
    return f"✅ Все готово! Используйте:\n" \
           f"   - GET /api/v1/recommend/{{user_id}}?n_recs=10 для получения рекомендаций\n" \
           f"   - GET /api/v1/metrics?k=10&sample_users=100 для оценки качества\n" \
           f"   - POST /api/v1/shap/initialize для SHAP интерпретации"
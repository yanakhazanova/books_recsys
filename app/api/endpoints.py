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

from app.services.shap_analyzer import SHAPAnalyzer
import random

from typing import Optional
import logging
logger = logging.getLogger(__name__)

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


@router.post("/train/continue", response_model=TrainResponse)
async def continue_training(
    additional_epochs: int = Query(5, ge=1, le=50, description="Дополнительных эпох"),
    learning_rate: Optional[float] = Query(None, ge=0.0001, le=0.01, description="Новая learning rate (опционально)"),
    reset_optimizer: bool = Query(False, description="Сбросить оптимизатор")
):
    """
    Продолжает обучение существующей модели (дообучение)
    """
    if state.model is None:
        raise HTTPException(status_code=400, detail="Модель не загружена. Сначала вызовите /train")
    
    try:
        # Загружаем фичи и триплеты
        if state.book_features is None:
            state.book_features, state.user_features, state.triplets = data_cache.load_features()
        
        if state.book_features is None or state.triplets is None:
            raise HTTPException(status_code=400, detail="Нет фичей. Сначала вызовите /features")
        
        # Продолжаем обучение
        state.model.continue_training(
            triplets_df=state.triplets,
            user_features=state.user_features,
            book_features=state.book_features,
            additional_epochs=additional_epochs,
            learning_rate=learning_rate,
            reset_optimizer=reset_optimizer
        )
        
        # Сохраняем обновленную модель
        state.model.save(str(settings.model_path))
        
        return TrainResponse(
            status="success",
            model_path=str(settings.model_path),
            cv_score=state.model.val_losses[-1] if state.model.val_losses else None,
            message=f"Добавлено {additional_epochs} эпох. Всего: {len(state.model.train_losses)}"
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


@router.post("/shap/initialize")
async def initialize_shap(
    n_background: int = Query(100, ge=10, le=500, description="Количество фоновых пар для SHAP"),
    n_test: int = Query(50, ge=10, le=200, description="Количество тестовых пар для глобального анализа")
):
    """
    Инициализирует SHAP объяснитель для глобальной интерпретации
    """    
    if state.model is None:
        raise HTTPException(status_code=400, detail="Модель не загружена. Сначала вызовите /train")
    
    if state.candidates is None:
        raise HTTPException(status_code=400, detail="Нет кандидатов. Сначала вызовите /candidates")
    
    try:
        # Собираем пары (user, book) из кандидатов
        all_pairs = []
        users = list(state.candidates.keys())
        
        # Для фона берем случайных пользователей и их кандидатов
        background_users = random.sample(users, min(n_background // 5, len(users)))
        for user_id in background_users:
            books = list(state.candidates[user_id].keys())[:5]  # по 5 книг на пользователя
            for book_id in books:
                all_pairs.append((user_id, book_id))
        
        # Создаем SHAP анализатор
        state.shap_analyzer = SHAPAnalyzer(
            model=state.model.model,
            trainer=state.model,
            device=state.model.device
        )
        
        # Инициализируем
        result = state.shap_analyzer.initialize(all_pairs[:n_background], n_samples=50)
        
        if result['status'] == 'error':
            raise HTTPException(status_code=400, detail=result['message'])
        
        return result
        
    except Exception as e:
        print(f"Ошибка инициализации SHAP: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/shap/global")
async def get_global_shap(
    n_pairs: int = Query(50, ge=10, le=200, description="Количество пар для глобального анализа"),
    nsamples_per_pair: int = Query(30, ge=10, le=100, description="Сэмплов SHAP на пару"),
    top_k: int = Query(20, ge=5, le=50, description="Количество топ-признаков для вывода")
):
    """
    Глобальная интерпретация признаков (на небольшом количестве примеров)
    """
    
    if state.shap_analyzer is None:
        raise HTTPException(status_code=400, detail="SHAP не инициализирован. Сначала вызовите POST /shap/initialize")
    
    if state.candidates is None:
        raise HTTPException(status_code=400, detail="Нет кандидатов")
    
    try:
        # Собираем случайные пары для анализа
        all_users = list(state.candidates.keys())
        test_pairs = []
        
        # Случайные пользователи и их книги
        sample_users = random.sample(all_users, min(n_pairs // 5, len(all_users)))
        for user_id in sample_users:
            books = list(state.candidates[user_id].keys())[:5]
            for book_id in books:
                test_pairs.append((user_id, book_id))
        
        # Вычисляем глобальную важность
        importance_df = state.shap_analyzer.get_global_importance(
            test_pairs=test_pairs,
            nsamples_per_pair=nsamples_per_pair,
            max_pairs=n_pairs
        )
        
        if importance_df.empty:
            return {
                'status': 'error',
                'message': 'Не удалось вычислить SHAP значения'
            }
        
        # Получаем описания признаков
        descriptions = state.shap_analyzer.get_feature_descriptions()
        
        # Формируем ответ
        top_features = importance_df.head(top_k)
        
        return {
            'status': 'success',
            'n_pairs_analyzed': len(test_pairs),
            'n_features_total': len(importance_df),
            'feature_importance': [
                {
                    'feature': row['feature'],
                    'description': descriptions.get(row['feature'], row['feature']),
                    'mean_abs_shap': float(row['mean_abs_shap']),
                    'mean_shap': float(row['mean_shap']),
                    'std_shap': float(row['std_shap'])
                }
                for _, row in top_features.iterrows()
            ],
            'all_features_summary': importance_df.to_dict(orient='records')
        }
        
    except Exception as e:
        print(f"Ошибка глобального SHAP: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/shap/explain/{user_id}/{book_id}")
async def explain_recommendation(
    user_id: int,
    book_id: str,
    nsamples: int = Query(100, ge=20, le=200, description="Количество сэмплов SHAP")
):
    """
    Объяснение рекомендации для конкретной пары (пользователь, книга)
    """    
    if state.shap_analyzer is None:
        raise HTTPException(status_code=400, detail="SHAP не инициализирован. Сначала вызовите POST /shap/initialize")
    
    try:
        # 🔧 ИСПРАВЛЕНО: Преобразуем book_id из индекса в ISBN если нужно
        actual_book_id = book_id
        
        # Проверяем, является ли book_id числовым индексом
        if book_id.isdigit() and hasattr(state.model, 'book_encoder'):
            # Пробуем восстановить ISBN по индексу
            book_idx = int(book_id)
            # Ищем книгу с таким индексом в book_encoder
            if hasattr(state.model, 'book_encoder') and hasattr(state.model.book_encoder, 'classes_'):
                if book_idx < len(state.model.book_encoder.classes_):
                    actual_book_id = state.model.book_encoder.classes_[book_idx]
                    print(f"🔍 Преобразован индекс {book_id} → ISBN {actual_book_id}")
        
        # Также проверяем, есть ли книга в book_features по индексу
        if hasattr(state.model, 'book_features') and state.model.book_features is not None:
            # Если book_id - это индекс и есть в индексах
            if book_id.isdigit() and int(book_id) in state.model.book_features.index:
                # Получаем ISBN из колонки если есть
                if 'ISBN' in state.model.book_features.columns:
                    actual_book_id = str(state.model.book_features.loc[int(book_id), 'ISBN'])
                    print(f"🔍 Найден ISBN {actual_book_id} для индекса {book_id}")
        
        # Получаем объяснение
        explanation = state.shap_analyzer.explain_prediction(user_id, actual_book_id, nsamples=nsamples)
        
        # 🔧 ИСПРАВЛЕНО: Проверяем, есть ли ошибка в объяснении
        if explanation is None:
            raise HTTPException(
                status_code=404, 
                detail=f"Не найдены данные для пользователя {user_id} или книги {book_id}"
            )
        
        if 'error' in explanation:
            raise HTTPException(
                status_code=400,
                detail=explanation['error']
            )
        
        # Получаем название книги
        book_title = None
        if state.books_titles_dict:
            book_title = state.books_titles_dict.get(str(actual_book_id))
        
        # Если не нашли по ISBN, пробуем найти по индексу
        if not book_title and hasattr(state.model, 'book_features') and state.model.book_features is not None:
            if 'Title' in state.model.book_features.columns:
                if book_id.isdigit() and int(book_id) in state.model.book_features.index:
                    book_title = state.model.book_features.loc[int(book_id), 'Title']
        
        # Получаем описания признаков
        descriptions = state.shap_analyzer.get_feature_descriptions()
        
        return {
            'status': 'success',
            'user_id': user_id,
            'book_id': actual_book_id,
            'original_book_id': book_id,  # Добавляем исходный ID для отладки
            'book_title': book_title,
            'predicted_score': explanation.get('predicted_score', 0),
            'shap_interpretation': {
                'top_positive_factors': [
                    {
                        'feature': feat,
                        'description': descriptions.get(feat, feat),
                        'shap_value': val
                    }
                    for feat, val in explanation.get('top_positive', [])[:5]
                ],
                'top_negative_factors': [
                    {
                        'feature': feat,
                        'description': descriptions.get(feat, feat),
                        'shap_value': val
                    }
                    for feat, val in explanation.get('top_negative', [])[:5]
                ]
            },
            'all_shap_values': explanation.get('shap_values', {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка объяснения: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/shap/explain/recommendations/{user_id}")
async def explain_user_recommendations(
    user_id: int,
    n_recs: int = Query(5, ge=1, le=10, description="Количество рекомендаций для объяснения"),
    nsamples: int = Query(50, ge=20, le=150, description="Сэмплов SHAP на рекомендацию")
):
    """
    Объяснение всех рекомендаций для пользователя
    """
    
    if state.shap_analyzer is None:
        raise HTTPException(status_code=400, detail="SHAP не инициализирован. Сначала вызовите POST /shap/initialize")
    
    if state.model is None:
        raise HTTPException(status_code=400, detail="Модель не загружена")
    
    try:
        # Получаем рекомендации для пользователя
        if user_id not in state.candidates:
            raise HTTPException(status_code=404, detail=f"Пользователь {user_id} не найден")
        
        candidate_books = list(state.candidates[user_id].keys())
        ranked = state.model.predict_for_user(user_id, candidate_books, top_n=n_recs)
        
        recommendations = []
        for book_id, score in ranked:
            # 🔧 ИСПРАВЛЕНО: Преобразуем book_id в ISBN если нужно
            actual_book_id = book_id
            
            # Пробуем найти ISBN
            if hasattr(state.model, 'book_features') and state.model.book_features is not None:
                # Если book_id - это строка, возможно уже ISBN
                # Проверяем, есть ли в индексе
                if book_id in state.model.book_features.index:
                    if 'ISBN' in state.model.book_features.columns:
                        actual_book_id = str(state.model.book_features.loc[book_id, 'ISBN'])
            
            # Получаем SHAP объяснение
            explanation = state.shap_analyzer.explain_prediction(user_id, actual_book_id, nsamples=nsamples)
            
            if explanation and 'error' not in explanation:
                book_title = None
                if state.books_titles_dict:
                    book_title = state.books_titles_dict.get(str(actual_book_id))
                
                recommendations.append({
                    'book_id': actual_book_id,
                    'original_book_id': book_id,  # Для отладки
                    'book_title': book_title,
                    'score': score,
                    'top_positive_factors': [
                        {'feature': feat, 'value': val} 
                        for feat, val in explanation.get('top_positive', [])[:3]
                    ],
                    'top_negative_factors': [
                        {'feature': feat, 'value': val} 
                        for feat, val in explanation.get('top_negative', [])[:3]
                    ]
                })
            else:
                # Если SHAP не сработал, добавляем без объяснения
                recommendations.append({
                    'book_id': book_id,
                    'book_title': state.books_titles_dict.get(str(book_id)) if state.books_titles_dict else None,
                    'score': score,
                    'top_positive_factors': [],
                    'top_negative_factors': [],
                    'shap_error': explanation.get('error') if explanation else 'Unknown error'
                })
        
        return {
            'status': 'success',
            'user_id': user_id,
            'n_recommendations': len(recommendations),
            'recommendations': recommendations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    

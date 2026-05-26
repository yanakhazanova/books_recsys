import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
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
from app.services.log_capture import get_log_path
from app.services.llm_explainer import (
    explain_recommendations as llm_explain_recommendations,
    LLMConfigError,
    DEFAULT_MODEL as LLM_DEFAULT_MODEL,
)
import random

from typing import Optional

router = APIRouter(prefix="/api/v1", tags=["recommendations"])


# --- ISBN -> [genres] lookup, lazily loaded once from the source pkl ---
_BOOK_GENRES_DICT = None
_GENRE_PKL_PATH = Path("data/raw/Books_with_genre_features.pkl")


def _get_book_genres_dict() -> dict:
    """ISBN(uppercased) -> list[str] of genre strings, built once from the
    same pkl that feeds add_genre_features. ~3 MB in memory.
    """
    global _BOOK_GENRES_DICT
    if _BOOK_GENRES_DICT is not None:
        return _BOOK_GENRES_DICT
    if not _GENRE_PKL_PATH.exists():
        _BOOK_GENRES_DICT = {}
        return _BOOK_GENRES_DICT
    try:
        import pandas as _pd
        print(f"📚 Загружаю genre lookup из {_GENRE_PKL_PATH}...")
        df = _pd.read_pickle(_GENRE_PKL_PATH)
        if 'isbn_10' not in df.columns or 'genres' not in df.columns:
            print("   В пкл нет колонок isbn_10/genres — пропускаю")
            _BOOK_GENRES_DICT = {}
            return _BOOK_GENRES_DICT
        d = {}
        for isbns, genres in zip(df['isbn_10'], df['genres']):
            if not isinstance(genres, (list, tuple)) or not genres:
                continue
            gl = [str(g) for g in genres]
            if isinstance(isbns, (list, tuple)):
                for isbn in isbns:
                    if isbn:
                        d[str(isbn).upper().strip()] = gl
            elif isbns:
                d[str(isbns).upper().strip()] = gl
        _BOOK_GENRES_DICT = d
        print(f"   ✅ Genre lookup: {len(d):,} ISBN → genres")
    except Exception as e:
        print(f"⚠️ Не удалось загрузить genre lookup: {e}")
        _BOOK_GENRES_DICT = {}
    return _BOOK_GENRES_DICT


@router.post("/prepare", response_model=PrepareDataResponse)
async def prepare_data():
    """Подготовка данных: загрузка, очистка, сохранение"""
    try:
        loader = DataLoader()
        result = loader.run_preparation()
        data_cache.save_cleaned_data(loader.ratings, loader.books, loader.users)

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
    """Генерация кандидатов коллаборативной фильтрацией + от авторов"""
    global state
    
    try:
        if state.train_ratings is None:
            state.train_ratings, state.test_ratings = data_cache.load_split()
        
        if state.train_ratings is None:
            raise HTTPException(status_code=400, detail="Данные не разделены. Сначала вызовите /split")
        
        if state.books_df is None:
            _, state.books_df, _ = data_cache.load_cleaned_data()
        
        # Генерируем кандидатов
        generator = CollaborativeGenerator(n_candidates=n_candidates)
        state.candidates = generator.generate(
            state.train_ratings, 
            state.test_ratings,
            state.books_df
        )
        
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
async def create_features(
      use_genre_features: bool = Query(True, description="Использовать жанровые эмбеддинги"),
      use_word2vec: bool = Query(True, description="Word2vec эмбеддинги жанров"),
      use_tfidf: bool = Query(False, description="TF-IDF эмбеддинги жанров"),
):
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
              users=users,
              use_genre_features=use_genre_features,
              use_word2vec=use_word2vec,
              use_tfidf=use_tfidf,
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
    
def _run_train_full_sync(test_items_per_user, n_candidates, use_existing_model,
                          epochs, batch_size, embed_dim, attention_dim,
                          learning_rate, skip_if_exists,
                          use_genre_features, use_word2vec, use_tfidf):
    """
    Запускает все стадии конвейера синхронно в отдельном event loop.
    Используется через asyncio.to_thread из train_full_pipeline, чтобы основной
    event loop оставался свободным для /logs/stream и других запросов.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        if not skip_if_exists or not data_cache.check_cleaned_data():
            loop.run_until_complete(prepare_data())
        if not skip_if_exists or not data_cache.check_split():
            loop.run_until_complete(
                split_train_test(test_items_per_user=test_items_per_user))
        if not skip_if_exists or not data_cache.check_candidates():
            loop.run_until_complete(
                generate_candidates(n_candidates=n_candidates))
        if not skip_if_exists or not data_cache.check_features():
              loop.run_until_complete(create_features(
                  use_genre_features=use_genre_features,
                  use_word2vec=use_word2vec,
                  use_tfidf=use_tfidf,
              ))
        return loop.run_until_complete(train_model(
            use_existing_model=use_existing_model,
            epochs=epochs,
            batch_size=batch_size,
            embed_dim=embed_dim,
            attention_dim=attention_dim,
            learning_rate=learning_rate,
        ))
    finally:
        try:
            loop.close()
        finally:
            asyncio.set_event_loop(None)


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
    skip_if_exists: bool = Query(False, description="Пропускать готовые этапы"),
    use_genre_features: bool = Query(True, description="Использовать жанровые эмбеддинги"),
    use_word2vec: bool = Query(True, description="Word2vec эмбеддинги жанров"),
    use_tfidf: bool = Query(False, description="TF-IDF эмбеддинги жанров"),
  ):
    """
    Полный пайплайн обучения одной командой.
    Тяжёлая работа уходит в отдельный поток, поэтому /logs/stream и другие
    эндпойнты продолжают отвечать пока модель обучается.
    """
    try:
        return await asyncio.to_thread(
              _run_train_full_sync,
              test_items_per_user, n_candidates, use_existing_model,
              epochs, batch_size, embed_dim, attention_dim, learning_rate,
              skip_if_exists,
              use_genre_features, use_word2vec, use_tfidf,
          )
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

@router.get("/shap/global")
async def get_global_shap(
    n_pairs: int = Query(20, ge=5, le=100, description="Количество пар для анализа"),
    nsamples_per_pair: int = Query(10, ge=5, le=50, description="Сэмплов SHAP на пару"),
    save_plot: bool = Query(True, description="Сохранять график")
):
    """
    Глобальная интерпретация признаков (автоматическая инициализация SHAP)
    """    
    if state.model is None:
        raise HTTPException(status_code=400, detail="Модель не загружена. Сначала вызовите /train")
    
    if state.user_features is None or state.book_features is None:
        raise HTTPException(status_code=400, detail="Фичи не загружены. Сначала вызовите /features")
    
    # Создаем анализатор если нужно
    if state.shap_analyzer is None:
        from app.services.shap_analyzer import SHAPAnalyzer
        
        # Убеждаемся, что в trainer есть данные
        if state.model.user_features is None:
            state.model.user_features = state.user_features
        if state.model.book_features is None:
            state.model.book_features = state.book_features
        
        state.shap_analyzer = SHAPAnalyzer(
            model=state.model.model,
            trainer=state.model,
            device=state.model.device,
            save_dir="models/shap"
        )
    
    try:
        # Собираем пары для анализа из реальных данных
        test_pairs = []
        
        if state.candidates:
            all_users = list(state.candidates.keys())
            if not all_users:
                raise HTTPException(status_code=400, detail="Нет пользователей с кандидатами")
            
            # Берем случайных пользователей
            sample_users = random.sample(all_users, min(max(1, n_pairs // 3), len(all_users)))
            
            for user_id in sample_users:
                books = list(state.candidates[user_id].keys())[:5]
                for book_id in books:
                    test_pairs.append((user_id, book_id))
                    if len(test_pairs) >= n_pairs:
                        break
                if len(test_pairs) >= n_pairs:
                    break
        
        # Если кандидатов нет, берем из book_features
        if not test_pairs and state.book_features is not None:
            users = list(state.user_features.index)[:min(10, len(state.user_features))]
            books = list(state.book_features.index)[:min(10, len(state.book_features))]
            
            for user_id in users:
                for book_id in books:
                    test_pairs.append((user_id, str(book_id)))
                    if len(test_pairs) >= n_pairs:
                        break
                if len(test_pairs) >= n_pairs:
                    break
        
        if not test_pairs:
            raise HTTPException(status_code=400, detail="Нет данных для анализа SHAP")
        
        # Получаем глобальную важность
        result = state.shap_analyzer.get_global_importance(
            test_pairs=test_pairs,
            nsamples_per_pair=nsamples_per_pair,
            max_pairs=n_pairs,
            save_plot=save_plot
        )
        
        if result.get('status') == 'error':
            raise HTTPException(status_code=400, detail=result.get('message', 'Ошибка SHAP'))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Ошибка глобального SHAP: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/shap/explain/{user_id}/{book_id}")
async def explain_recommendation(
    user_id: int,
    book_id: str,
    nsamples: int = Query(50, ge=10, le=100, description="Сэмплов SHAP"),
    save_plot: bool = Query(True, description="Сохранять график")
):
    """
    Объяснение рекомендации для конкретной пары (автоматическая инициализация)
    """
    
    if state.model is None:
        raise HTTPException(status_code=400, detail="Модель не загружена. Сначала вызовите /train")
    
    if state.user_features is None or state.book_features is None:
        raise HTTPException(status_code=400, detail="Фичи не загружены. Сначала вызовите /features")
    
    # Создаем анализатор если нужно
    if state.shap_analyzer is None:
        from app.services.shap_analyzer import SHAPAnalyzer
        
        if state.model.user_features is None:
            state.model.user_features = state.user_features
        if state.model.book_features is None:
            state.model.book_features = state.book_features
        
        state.shap_analyzer = SHAPAnalyzer(
            model=state.model.model,
            trainer=state.model,
            device=state.model.device,
            save_dir="models/shap"
        )
    
    try:
        # Преобразуем book_id если нужно
        actual_book_id = book_id
        
        # Если book_id - это число (индекс), пытаемся найти ISBN
        if book_id.isdigit() and state.book_features is not None:
            idx = int(book_id)
            if idx < len(state.book_features.index):
                actual_book_id = str(state.book_features.index[idx])
        
        # Получаем объяснение
        explanation = state.shap_analyzer.explain_prediction(
            user_id, actual_book_id, nsamples=nsamples, save_plot=save_plot
        )
        
        if 'error' in explanation:
            raise HTTPException(status_code=400, detail=explanation['error'])
        
        # Получаем название книги
        book_title = None
        if state.books_titles_dict:
            book_title = state.books_titles_dict.get(str(actual_book_id))
        
        descriptions = state.shap_analyzer.get_feature_descriptions()
        
        return {
            'status': 'success',
            'user_id': user_id,
            'book_id': actual_book_id,
            'original_book_id': book_id,
            'book_title': book_title,
            'predicted_score': explanation['predicted_score'],
            'plot_path': explanation.get('plot_path'),
            'shap_interpretation': {
                'top_positive_factors': [
                    {'feature': feat, 'description': descriptions.get(feat, feat), 'shap_value': val}
                    for feat, val in explanation['top_positive'][:5]
                ],
                'top_negative_factors': [
                    {'feature': feat, 'description': descriptions.get(feat, feat), 'shap_value': val}
                    for feat, val in explanation['top_negative'][:5]
                ]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Ошибка объяснения: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/shap/explain/recommendations/{user_id}")
async def explain_user_recommendations(
    user_id: int,
    n_recs: int = Query(5, ge=1, le=10, description="Количество рекомендаций"),
    nsamples: int = Query(30, ge=10, le=80, description="Сэмплов SHAP на рекомендацию"),
    save_plot: bool = Query(True, description="Сохранять график")
):
    """
    Объяснение всех рекомендаций для пользователя (автоматическая инициализация)
    """
    
    if state.model is None:
        raise HTTPException(status_code=400, detail="Модель не загружена. Сначала вызовите /train")
    
    if state.user_features is None or state.book_features is None:
        raise HTTPException(status_code=400, detail="Фичи не загружены. Сначала вызовите /features")
    
    # Создаем анализатор если нужно
    if state.shap_analyzer is None:
        from app.services.shap_analyzer import SHAPAnalyzer
        
        if state.model.user_features is None:
            state.model.user_features = state.user_features
        if state.model.book_features is None:
            state.model.book_features = state.book_features
        
        state.shap_analyzer = SHAPAnalyzer(
            model=state.model.model,
            trainer=state.model,
            device=state.model.device,
            save_dir="models/shap"
        )
    
    try:
        if state.candidates is None:
            raise HTTPException(status_code=400, detail="Нет кандидатов. Сначала вызовите /candidates")
        
        if user_id not in state.candidates:
            raise HTTPException(status_code=404, detail=f"Пользователь {user_id} не найден в кандидатах")
        
        candidate_books = list(state.candidates[user_id].keys())
        if not candidate_books:
            return {
                'status': 'success',
                'user_id': user_id,
                'recommendations': [],
                'message': 'Нет книг-кандидатов для этого пользователя'
            }
        
        ranked = state.model.predict_for_user(user_id, candidate_books, top_n=n_recs)
        
        if not ranked:
            return {
                'status': 'success',
                'user_id': user_id,
                'recommendations': [],
                'message': 'Модель не выдала рекомендаций для этого пользователя'
            }
        
        # Получаем объяснения
        result = state.shap_analyzer.explain_user_recommendations(
            user_id=user_id,
            recommendations=ranked,
            nsamples=nsamples,
            save_plot=save_plot
        )
        
        # Добавляем названия книг + жанры (для человекочитаемого декодинга)
        genres_lookup = _get_book_genres_dict()
        for rec in result['recommendations']:
            book_id_str = str(rec['book_id'])
            book_title = None
            if state.books_titles_dict:
                book_title = state.books_titles_dict.get(book_id_str)
            rec['book_title'] = book_title
            rec['genres'] = genres_lookup.get(book_id_str.upper().strip(), [])
        
        return {
            'status': 'success',
            'user_id': user_id,
            'plot_path': result.get('plot_path'),
            'recommendations': result['recommendations']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Ошибка: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
#  LLM-powered explanations (OpenRouter)
# ============================================================

def _build_user_profile(user_id: int) -> dict:
    """Собрать профиль пользователя из state.train_ratings + genre lookup."""
    profile = {}
    tr = state.train_ratings
    if tr is None:
        return profile
    user_rows = tr[tr['User-ID'] == user_id]
    if not len(user_rows):
        return profile
    nonzero = user_rows[user_rows['Rating'] > 0]
    profile['rating_count'] = int(len(user_rows))
    profile['non_zero_ratio'] = float(len(nonzero) / max(len(user_rows), 1))
    if len(nonzero):
        profile['avg_rating'] = float(nonzero['Rating'].mean())

    # Топ-жанры, которые этот пользователь читал (из его train-ISBN)
    genres_lookup = _get_book_genres_dict()
    if genres_lookup:
        from collections import Counter
        counter: Counter = Counter()
        for isbn in user_rows['ISBN'].astype(str).str.upper().str.strip().tolist():
            for g in genres_lookup.get(isbn, []):
                counter[g] += 1
        if counter:
            profile['top_genres'] = [g for g, _ in counter.most_common(8)]
    return profile


@router.post("/explain/llm/{user_id}")
async def explain_with_llm(
    user_id: int,
    n_recs: int = Query(5, ge=1, le=10, description="Сколько рекомендаций объяснять"),
    nsamples: int = Query(30, ge=10, le=80, description="SHAP-сэмплов на рекомендацию"),
    model: str = Query(LLM_DEFAULT_MODEL, description="OpenRouter model slug"),
):
    """
    Запрашивает у LLM (через OpenRouter) человекочитаемые объяснения
    того, почему модель порекомендовала эти книги. Возвращает текст
    на каждую книгу. Требует OPENROUTER_API_KEY в env.
    """
    # Reuse the SHAP explanation path
    if state.model is None:
        raise HTTPException(status_code=400, detail="Модель не загружена. Сначала /train")
    if state.user_features is None or state.book_features is None:
        raise HTTPException(status_code=400, detail="Фичи не загружены. Сначала /features")
    if state.candidates is None:
        state.candidates = data_cache.load_candidates()
        if state.candidates is None:
            raise HTTPException(status_code=400, detail="Нет кандидатов. Сначала /candidates")
    if user_id not in state.candidates:
        raise HTTPException(
            status_code=404,
            detail=f"Пользователь {user_id} не найден в кандидатах "
                    f"(после очистки осталось {len(state.candidates)} пользователей)."
        )

    if state.shap_analyzer is None:
        if state.model.user_features is None:
            state.model.user_features = state.user_features
        if state.model.book_features is None:
            state.model.book_features = state.book_features
        state.shap_analyzer = SHAPAnalyzer(
            model=state.model.model,
            trainer=state.model,
            device=state.model.device,
            save_dir="models/shap",
        )

    try:
        candidate_books = list(state.candidates[user_id].keys())
        ranked = state.model.predict_for_user(user_id, candidate_books, top_n=n_recs)
        if not ranked:
            return {"status": "success", "user_id": user_id,
                    "recommendations": [], "model": model}

        shap_result = state.shap_analyzer.explain_user_recommendations(
            user_id=user_id, recommendations=ranked, nsamples=nsamples,
            save_plot=False,
        )

        # Enrich recommendations with titles + genres (same as /shap/explain/recommendations)
        genres_lookup = _get_book_genres_dict()
        recs = shap_result.get('recommendations', [])
        for rec in recs:
            book_id_str = str(rec['book_id'])
            if state.books_titles_dict:
                rec['book_title'] = state.books_titles_dict.get(book_id_str)
            rec['genres'] = genres_lookup.get(book_id_str.upper().strip(), [])

        user_profile = _build_user_profile(user_id)

        # Call OpenRouter
        llm_result = await llm_explain_recommendations(
            user_profile=user_profile,
            recommendations=recs,
            model=model,
        )
        explanations = llm_result.get('explanations', {})

        for rec in recs:
            bid = str(rec['book_id'])
            rec['llm_explanation'] = (
                explanations.get(bid)
                or explanations.get('_global')
                or ''
            )

        return {
            "status": "success",
            "user_id": user_id,
            "user_profile": user_profile,
            "model": llm_result.get('model', model),
            "tokens": llm_result.get('tokens', {}),
            "recommendations": recs,
        }

    except LLMConfigError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ LLM explain failed: {e}")
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
#  Live server logs (Server-Sent Events)
# ============================================================

def _sse(line: str) -> str:
    """Один блок SSE (data: ...). Многострочный текст разбивается на data: lines."""
    safe = line.replace('\r', '')
    parts = [f"data: {chunk}" for chunk in safe.split('\n')]
    return "\n".join(parts) + "\n\n"


@router.get("/logs/stream")
async def stream_logs(
    tail: int = Query(300, ge=0, le=2000,
                       description="Сколько последних строк отдать при подключении"),
):
    """
    Стрим логов сервера через Server-Sent Events. Сначала отдаёт последние
    `tail` строк, потом подписывается на новые записи в файле и отдаёт их
    как только они появляются.
    """
    log_path = get_log_path()
    if log_path is None or not log_path.exists():
        raise HTTPException(status_code=503, detail="Log capture не включён")

    async def gen():
        # ---- начальный хвост ----
        try:
            with open(log_path, 'rb') as f:
                f.seek(0, 2)
                size = f.tell()
                read_back = min(size, 256 * 1024)  # не больше 256 KB при первом чтении
                f.seek(size - read_back)
                buf = f.read(read_back).decode('utf-8', errors='replace')
            if size > read_back:
                # отрезаем возможную обрезанную первую строку
                nl = buf.find('\n')
                if nl != -1:
                    buf = buf[nl + 1:]
            lines = buf.splitlines()[-tail:] if tail else []
            for line in lines:
                yield _sse(line)
        except FileNotFoundError:
            yield _sse("[лог-файл ещё не создан]")
            return

        # ---- ожидание новых строк ----
        # Открываем файл повторно и сидим в конце. На каждой итерации даём
        # шанс циклу выйти из ожидания (cancellation/client disconnect).
        try:
            with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                f.seek(0, 2)
                idle = 0
                while True:
                    line = f.readline()
                    if line:
                        yield _sse(line.rstrip('\n'))
                        idle = 0
                    else:
                        await asyncio.sleep(0.4)
                        idle += 1
                        # heartbeat каждые ~8s, чтобы прокси не убил соединение
                        if idle >= 20:
                            yield ": keep-alive\n\n"
                            idle = 0
        except asyncio.CancelledError:
            raise
        except Exception as e:
            yield _sse(f"[stream error: {e}]")

    return StreamingResponse(
        gen(),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        },
    )

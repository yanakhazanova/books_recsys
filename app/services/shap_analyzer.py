# app/services/shap_analyzer.py
import shap
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """Класс для SHAP интерпретации рекомендаций с автоматической инициализацией"""
    
    def __init__(self, model, trainer, device='cpu', save_dir="models/shap"):
        """
        Args:
            model: AFMRanker модель
            trainer: AFMTrainer с энкодерами и скейлерами
            device: 'cpu' или 'cuda'
            save_dir: директория для сохранения графиков
        """
        self.model = model
        self.trainer = trainer
        self.device = device
        self.explainer = None
        self.background_data = None
        self.feature_names = None
        self.is_initialized = False
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Определяем признаки
        self._define_features()
    
    def _define_features(self):
        """Определяет список всех признаков для SHAP"""
        self.user_num_features = []
        self.book_num_features = []
        
        # Берем ВСЕ числовые колонки из trainer
        if hasattr(self.trainer, 'user_num_cols') and self.trainer.user_num_cols:
            self.user_num_features = self.trainer.user_num_cols
        elif hasattr(self.trainer, 'user_features') and self.trainer.user_features is not None:
            # Все числовые колонки из user_features
            self.user_num_features = self.trainer.user_features.select_dtypes(include=[np.number]).columns.tolist()
            # Убираем индекс если есть
            self.user_num_features = [col for col in self.user_num_features if col not in ['User-ID', 'user_id', 'index']]
        
        if hasattr(self.trainer, 'book_num_cols') and self.trainer.book_num_cols:
            self.book_num_features = self.trainer.book_num_cols
        elif hasattr(self.trainer, 'book_features') and self.trainer.book_features is not None:
            self.book_num_features = self.trainer.book_features.select_dtypes(include=[np.number]).columns.tolist()
            self.book_num_features = [col for col in self.book_num_features if col not in ['ISBN', 'index']]
        
        # Формируем имена признаков
        self.feature_names = (
            [f'user_{col}' for col in self.user_num_features] +
            [f'book_{col}' for col in self.book_num_features]
        )
        
        print(f"✅ SHAP: Определено {len(self.feature_names)} признаков")
        print(f"   User: {len(self.user_num_features)}")
        print(f"   Book: {len(self.book_num_features)}")
        if len(self.feature_names) > 0:
            print(f"   Примеры: {self.feature_names[:5]}...")

    def _get_default_background_pairs(self, n_users: int = 20, n_books_per_user: int = 5) -> List[Tuple[int, str]]:
        """Получает дефолтные пары для инициализации"""
        pairs = []
        
        if not hasattr(self.trainer, 'user_features') or self.trainer.user_features is None:
            return pairs
        
        if not hasattr(self.trainer, 'book_features') or self.trainer.book_features is None:
            return pairs
        
        users = list(self.trainer.user_features.index)[:n_users]
        
        for user_id in users:
            books = list(self.trainer.book_features.index)[:n_books_per_user]
            for book_id in books:
                pairs.append((user_id, str(book_id)))
        
        return pairs
    
    def _ensure_initialized(self, background_pairs: List[Tuple[int, str]] = None, force_reinit: bool = False) -> bool:
        """
        Автоматическая инициализация SHAP если нужно
        
        Returns:
            bool: успешна ли инициализация
        """
        if self.is_initialized and not force_reinit:
            logger.info("SHAP уже инициализирован")
            return True
        
        logger.info("Автоматическая инициализация SHAP...")
        
        if background_pairs is None:
            background_pairs = self._get_default_background_pairs()
        
        if not background_pairs:
            logger.error("Нет доступных пар для инициализации")
            return False
        
        background_vectors = []
        for user_id, book_id in background_pairs:
            vec = self._build_feature_vector(user_id, book_id)
            if vec is not None:
                background_vectors.append(vec)
        
        if len(background_vectors) < 10:
            logger.warning(f"Недостаточно фоновых пар: {len(background_vectors)}")
            return False
        
        self.background_data = np.array(background_vectors)
        
        try:
            self.explainer = shap.KernelExplainer(self._predict_fn, self.background_data)
            self.is_initialized = True
            logger.info(f"SHAP инициализирован на {len(background_vectors)} парах")
            return True
        except Exception as e:
            logger.error(f"Ошибка инициализации: {e}")
            return False
    
    def _build_feature_vector(self, user_id: int, book_id: str) -> Optional[np.ndarray]:
        """Строит вектор признаков для пары (user, book)"""
        
        # Проверяем доступность данных
        if not hasattr(self.trainer, 'user_features') or self.trainer.user_features is None:
            print(f"   ❌ user_features не загружены в trainer")
            return None
        
        if not hasattr(self.trainer, 'book_features') or self.trainer.book_features is None:
            print(f"   ❌ book_features не загружены в trainer")
            return None
        
        # Проверяем пользователя
        if user_id not in self.trainer.user_features.index:
            # Пробуем преобразовать тип
            try:
                user_id_int = int(user_id)
                if user_id_int in self.trainer.user_features.index:
                    user_id = user_id_int
                else:
                    print(f"   ❌ Пользователь {user_id} не найден в user_features")
                    return None
            except:
                print(f"   ❌ Пользователь {user_id} не найден в user_features")
                return None
        
        # Проверяем книгу
        book_id_str = str(book_id)
        book_row = None
        
        # Ищем в индексе
        if book_id_str in self.trainer.book_features.index:
            book_row = self.trainer.book_features.loc[book_id_str]
        # Ищем по ISBN колонке
        elif 'ISBN' in self.trainer.book_features.columns:
            mask = self.trainer.book_features['ISBN'].astype(str) == book_id_str
            if mask.any():
                book_row = self.trainer.book_features[mask].iloc[0]
            else:
                # Пробуем искать по числовому индексу
                try:
                    book_id_int = int(book_id_str)
                    if book_id_int in self.trainer.book_features.index:
                        book_row = self.trainer.book_features.loc[book_id_int]
                except:
                    pass
        
        if book_row is None:
            print(f"   ❌ Книга {book_id_str} не найдена в book_features")
            return None
        
        user_f = self.trainer.user_features.loc[user_id]
        
        # Строим вектор
        user_num = []
        for col in self.user_num_features:
            if col in user_f.index:
                val = user_f[col]
                if pd.isna(val):
                    val = 0.0
                if hasattr(val, 'iloc'):
                    val = val.iloc[0] if len(val) > 0 else 0.0
                user_num.append(float(val))
            else:
                user_num.append(0.0)
        
        book_num = []
        for col in self.book_num_features:
            if col in book_row.index:
                val = book_row[col]
                if pd.isna(val):
                    val = 0.0
                if hasattr(val, 'iloc'):
                    val = val.iloc[0] if len(val) > 0 else 0.0
                book_num.append(float(val))
            else:
                book_num.append(0.0)

        # ВАЖНО: применяем StandardScaler точно так же, как это делает
        # AFMDataset.__getitem__ при обучении. Иначе модель видит сырые
        # значения (e.g. publisher_book_count=2871), интерпретирует их как
        # z-score и выдаёт абсурдные предсказания → SHAP-значения в сотни.
        user_num_arr = np.asarray(user_num, dtype=np.float32).reshape(1, -1)
        book_num_arr = np.asarray(book_num, dtype=np.float32).reshape(1, -1)
        if (getattr(self.trainer, 'user_scaler', None) is not None
                and user_num_arr.size > 0):
            try:
                user_num_arr = self.trainer.user_scaler.transform(user_num_arr)
            except Exception as e:
                logger.warning(f"user_scaler.transform не сработал, использую сырые значения: {e}")
        if (getattr(self.trainer, 'book_scaler', None) is not None
                and book_num_arr.size > 0):
            try:
                book_num_arr = self.trainer.book_scaler.transform(book_num_arr)
            except Exception as e:
                logger.warning(f"book_scaler.transform не сработал, использую сырые значения: {e}")

        vector = np.concatenate([user_num_arr.flatten(), book_num_arr.flatten()])

        if vector.size == 0:
            print(f"   ❌ Пустой вектор для user={user_id}, book={book_id_str}")
            return None
        # print(f"   ✅ Вектор построен: {len(vector)} признаков")
        return vector.astype(np.float32)

    def _predict_fn(self, x: np.ndarray) -> np.ndarray:
        """Функция предсказания для SHAP"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(x.shape[0]):
                try:
                    n_user_num = len(self.user_num_features)
                    n_book_num = len(self.book_num_features)
                    
                    user_num_arr = x[i, :n_user_num] if n_user_num > 0 else np.array([])
                    book_num_arr = x[i, n_user_num:n_user_num + n_book_num] if n_book_num > 0 else np.array([])
                    
                    user_tensor = torch.tensor([0], device=self.device)
                    book_tensor = torch.tensor([0], device=self.device)
                    
                    user_num = torch.tensor(user_num_arr, dtype=torch.float32, device=self.device).unsqueeze(0) if len(user_num_arr) > 0 else None
                    book_num = torch.tensor(book_num_arr, dtype=torch.float32, device=self.device).unsqueeze(0) if len(book_num_arr) > 0 else None
                    
                    score = self.model.forward_single(user_tensor, book_tensor, user_num, book_num)
                    predictions.append(score.item())
                except Exception as e:
                    predictions.append(0.0)
        
        return np.array(predictions)
    
    @staticmethod
    def _aggregate_shap_array(names: List[str], matrix: np.ndarray,
                               aggregate: bool = True) -> Tuple[List[str], np.ndarray]:
        """
        Объединяет 64 столбца `genre_emb_*` (и `genre_tfidf_emb_*`) в один
        столбец signed-sum. SHAP — аддитивная мера (efficiency axiom):
        сумма phi_i по группе = вклад группы в (f(x) - E[f(X)]).
        Поэтому такая склейка строго эквивалентна "вкладу группы" и не
        теряет информации.

        Args:
            names:  список имён фичей длины M
            matrix: SHAP-массив формы (n_samples, M)  или (M,)
            aggregate: если False — no-op (возвращает копию)
        Returns: (new_names, new_matrix), где M уменьшается.
        """
        if not aggregate:
            return list(names), np.array(matrix, copy=True)

        m = np.atleast_2d(np.asarray(matrix, dtype=np.float32))

        keep_idx, w2v_idx, tfidf_idx = [], [], []
        for j, name in enumerate(names):
            parts = name.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                stem = parts[0]
                if stem.endswith('genre_tfidf_emb'):
                    tfidf_idx.append(j); continue
                if stem.endswith('genre_emb'):
                    w2v_idx.append(j); continue
            keep_idx.append(j)

        new_names = [names[j] for j in keep_idx]
        chunks = [m[:, keep_idx]] if keep_idx else []
        if w2v_idx:
            new_names.append('book_genre_w2v_total')
            chunks.append(m[:, w2v_idx].sum(axis=1, keepdims=True))
        if tfidf_idx:
            new_names.append('book_genre_tfidf_total')
            chunks.append(m[:, tfidf_idx].sum(axis=1, keepdims=True))

        new_matrix = np.concatenate(chunks, axis=1) if chunks else m
        # Squeeze back if input was 1D
        if matrix.ndim == 1:
            new_matrix = new_matrix[0]
        return new_names, new_matrix

    def get_global_importance(self, test_pairs: List[Tuple[int, str]],
                              nsamples_per_pair: int = 50,
                              max_pairs: int = 100,
                              save_plot: bool = True,
                              aggregate_groups: bool = True) -> Dict:
        """
        Вычисляет глобальную важность признаков и сохраняет график
        """
        if not self._ensure_initialized():
            return {'status': 'error', 'message': 'Не удалось инициализировать SHAP'}
        
        test_pairs = test_pairs[:max_pairs]
        
        test_vectors = []
        for user_id, book_id in test_pairs:
            vec = self._build_feature_vector(user_id, book_id)
            if vec is not None:
                test_vectors.append(vec)
        
        if len(test_vectors) == 0:
            return {'status': 'error', 'message': 'Нет валидных тестовых пар'}
        
        X_test = np.array(test_vectors)
        
        logger.info(f"Вычисление SHAP для {len(test_vectors)} тестовых пар...")
        
        all_shap_values = []
        for i in tqdm(range(len(X_test)), desc="SHAP computation"):
            try:
                shap_vals = self.explainer.shap_values(X_test[i:i+1], nsamples=nsamples_per_pair)
                
                if len(shap_vals.shape) == 3:
                    shap_1d = shap_vals[0, :, 0]
                elif len(shap_vals.shape) == 2:
                    shap_1d = shap_vals[0, :]
                else:
                    shap_1d = shap_vals[0]
                
                all_shap_values.append(shap_1d)
            except Exception as e:
                all_shap_values.append(np.zeros(len(self.feature_names)))
        
        if not all_shap_values:
            return {'status': 'error', 'message': 'Не удалось вычислить SHAP значения'}
        
        shap_matrix = np.array(all_shap_values)

        # Сворачиваем 64 жанровых эмбеддинга в один столбец genre_w2v_total.
        # SHAP-аддитивность гарантирует: signed sum по группе = вклад группы.
        agg_names, shap_matrix = self._aggregate_shap_array(
            self.feature_names, shap_matrix, aggregate=aggregate_groups
        )

        mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
        mean_shap = shap_matrix.mean(axis=0)
        std_shap = shap_matrix.std(axis=0)

        importance_df = pd.DataFrame({
            'feature': agg_names,
            'mean_abs_shap': mean_abs_shap,
            'mean_shap': mean_shap,
            'std_shap': std_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        plot_path = None
        if save_plot:
            plot_path = self._save_global_importance_plot(importance_df)
        
        return {
            'status': 'success',
            'n_pairs_analyzed': len(test_vectors),
            'n_features_total': len(self.feature_names),
            'feature_importance': importance_df.head(20).to_dict(orient='records'),
            'plot_path': plot_path
        }
    
    def _save_global_importance_plot(self, importance_df: pd.DataFrame) -> str:
        """Сохраняет график глобальной важности признаков"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.save_dir / f"global_importance_{timestamp}.png"
        
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(15)
        
        bars = plt.barh(range(len(top_features)), top_features['mean_abs_shap'], color='steelblue')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Mean |SHAP| value', fontsize=12)
        plt.title('Global Feature Importance (SHAP)', fontsize=14)
        plt.tight_layout()
        
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"График сохранен: {plot_path}")
        return str(plot_path)
    
    def explain_prediction(self, user_id: int, book_id: str, nsamples: int = 100,
                            save_plot: bool = True, aggregate_groups: bool = True) -> Dict:
        """
        Объясняет предсказание для конкретной пары и сохраняет график
        """
        if not self._ensure_initialized():
            return {'error': 'Не удалось инициализировать SHAP'}
        
        feature_vector = self._build_feature_vector(user_id, book_id)
        if feature_vector is None:
            return {'error': f'Не удалось построить вектор для user={user_id}, book={book_id}'}
        
        try:
            shap_values = self.explainer.shap_values(feature_vector.reshape(1, -1), nsamples=nsamples)
            
            if len(shap_values.shape) == 3:
                shap_1d = shap_values[0, :, 0]
            elif len(shap_values.shape) == 2:
                shap_1d = shap_values[0, :]
            else:
                shap_1d = shap_values[0]
            
            # Сворачиваем эмбеддинговые группы (см. SHAP additivity proof)
            agg_names, agg_vals = self._aggregate_shap_array(
                self.feature_names, shap_1d, aggregate=aggregate_groups
            )
            explanation = {
                agg_names[i]: float(agg_vals[i])
                for i in range(len(agg_names))
            }
            
            score = self._predict_fn(feature_vector.reshape(1, -1))[0]
            
            top_positive = sorted([(k, v) for k, v in explanation.items() if v > 0], 
                                  key=lambda x: x[1], reverse=True)[:10]
            top_negative = sorted([(k, v) for k, v in explanation.items() if v < 0], 
                                  key=lambda x: x[1])[:10]
            
            plot_path = None
            if save_plot:
                plot_path = self._save_local_explanation_plot(
                    user_id, book_id, top_positive, top_negative, score
                )
            
            return {
                'user_id': user_id,
                'book_id': book_id,
                'predicted_score': float(score),
                'shap_values': explanation,
                'top_positive': top_positive,
                'top_negative': top_negative,
                'plot_path': plot_path
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _save_local_explanation_plot(self, user_id: int, book_id: str, 
                                      top_positive: List, top_negative: List, 
                                      score: float) -> str:
        """Сохраняет график локального объяснения"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.save_dir / f"local_explanation_user_{user_id}_book_{book_id}_{timestamp}.png"
        
        all_top = top_positive[:5] + top_negative[:5]
        features = [f[0] for f in all_top]
        values = [f[1] for f in all_top]
        colors = ['green' if v > 0 else 'red' for v in values]
        
        plt.figure(figsize=(10, 6))
        plt.barh(features, values, color=colors)
        plt.xlabel('SHAP value', fontsize=12)
        plt.title(f'SHAP Explanation for User {user_id}\nBook: {book_id} | Score: {score:.4f}', fontsize=12)
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def explain_user_recommendations(self, user_id: int, recommendations: List[Tuple[str, float]], 
                                      nsamples: int = 50, save_plot: bool = True) -> Dict:
        """
        Объясняет все рекомендации для пользователя и сохраняет сводный график
        """
        results = []
        
        for book_id, score in recommendations[:5]:
            explanation = self.explain_prediction(user_id, book_id, nsamples, save_plot=False)

            self.diagnose_feature_mapping(user_id, book_id)
            
            if 'error' not in explanation:
                results.append({
                    'book_id': book_id,
                    'score': score,
                    'top_positive': explanation['top_positive'],
                    'top_negative': explanation['top_negative']
                })
        
        plot_path = None
        if save_plot and results:
            plot_path = self._save_user_summary_plot(user_id, results)
        
        return {
            'user_id': user_id,
            'n_recommendations': len(results),
            'recommendations': results,
            'plot_path': plot_path
        }
    
    def _save_user_summary_plot(self, user_id: int, results: List) -> str:
        """Сохраняет сводный график для всех рекомендаций пользователя"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.save_dir / f"user_{user_id}_recommendations_{timestamp}.png"
        
        n_recs = len(results)
        fig, axes = plt.subplots(n_recs, 1, figsize=(12, 5 * n_recs))
        
        if n_recs == 1:
            axes = [axes]
        
        for i, rec in enumerate(results):
            book_id = rec['book_id'][:30] if len(rec['book_id']) > 30 else rec['book_id']
            score = rec['score']
            
            features = []
            values = []
            
            for feat, val in rec['top_positive'][:3]:
                features.append(feat)
                values.append(val)
            for feat, val in rec['top_negative'][:3]:
                features.append(feat)
                values.append(val)
            
            colors = ['green' if v > 0 else 'red' for v in values]
            
            axes[i].barh(features, values, color=colors)
            axes[i].set_xlabel('SHAP value', fontsize=10)
            axes[i].set_title(f'Book: {book_id}... (score: {score:.4f})', fontsize=11)
            axes[i].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.suptitle(f'SHAP Explanations for User {user_id}', fontsize=14)
        plt.tight_layout()
        
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Возвращает описания признаков"""
        descriptions = {}
        
        for col in self.user_num_features:
            descriptions[f'user_{col}'] = f'Признак пользователя: {col}'
        
        for col in self.book_num_features:
            descriptions[f'book_{col}'] = f'Признак книги: {col}'
        
        return descriptions
    
    def diagnose_feature_mapping(self, user_id: int, book_id: str):
        """
        Диагностика: что реально попадает в SHAP
        """
        print("=" * 60)
        print("ДИАГНОСТИКА SHAP ANALYZER")
        print("=" * 60)
        
        # 1. Какие признаки определены
        print(f"\n1. ОПРЕДЕЛЕННЫЕ ПРИЗНАКИ:")
        print(f"   User numerical: {self.user_num_features}")
        print(f"   Book numerical: {self.book_num_features}")
        print(f"   Всего признаков в SHAP: {len(self.feature_names)}")
        
        # 2. Проверяем trainer
        print(f"\n2. ДАННЫЕ В TRAINER:")
        print(f"   trainer.user_features exists: {hasattr(self.trainer, 'user_features') and self.trainer.user_features is not None}")
        print(f"   trainer.book_features exists: {hasattr(self.trainer, 'book_features') and self.trainer.book_features is not None}")
        
        if hasattr(self.trainer, 'user_features') and self.trainer.user_features is not None:
            print(f"   user_features shape: {self.trainer.user_features.shape}")
            print(f"   user_features columns: {list(self.trainer.user_features.columns)[:10]}")
        
        if hasattr(self.trainer, 'book_features') and self.trainer.book_features is not None:
            print(f"   book_features shape: {self.trainer.book_features.shape}")
            print(f"   book_features columns: {list(self.trainer.book_features.columns)[:10]}")
        
        # 3. Пробуем построить вектор для конкретной пары
        print(f"\n3. ПОСТРОЕНИЕ ВЕКТОРА ДЛЯ user={user_id}, book={book_id}:")
        vec = self._build_feature_vector(user_id, book_id)
        
        if vec is not None:
            print(f"   Вектор построен, размерность: {len(vec)}")
            print(f"   Не нулевых признаков: {np.sum(vec != 0)}")
            print(f"   Первые 10 значений: {vec[:10]}")
            
            # Сравниваем с ожидаемой размерностью
            expected_len = len(self.user_num_features) + len(self.book_num_features)
            print(f"   Ожидаемая размерность: {expected_len}")
            
            if len(vec) != expected_len:
                print(f"   ❌ РАЗМЕРНОСТЬ НЕ СОВПАДАЕТ!")
        else:
            print(f"   ❌ Не удалось построить вектор")
        
        # 4. Проверяем, какие признаки реально используются в модели
        print(f"\n4. МОДЕЛЬ AFM:")
        if hasattr(self.model, 'user_embedding'):
            print(f"   user_embedding size: {self.model.user_embedding.num_embeddings} x {self.model.user_embedding.embedding_dim}")
        if hasattr(self.model, 'book_embedding'):
            print(f"   book_embedding size: {self.model.book_embedding.num_embeddings} x {self.model.book_embedding.embedding_dim}")
        
        # 5. Проверяем, какие фичи реально передаются в модель при forward_single
        print(f"\n5. ПРОВЕРКА forward_single:")
        try:
            # Получаем индексы
            user_idx = self.trainer.user_encoder.transform([user_id])[0] if hasattr(self.trainer, 'user_encoder') else 0
            book_idx = self.trainer.book_encoder.transform([book_id])[0] if hasattr(self.trainer, 'book_encoder') else 0
            
            # Строим тензоры
            user_num = torch.tensor(vec[:len(self.user_num_features)], dtype=torch.float32).unsqueeze(0) if len(self.user_num_features) > 0 else None
            book_num = torch.tensor(vec[len(self.user_num_features):], dtype=torch.float32).unsqueeze(0) if len(self.book_num_features) > 0 else None
            
            print(f"   user_idx: {user_idx}, book_idx: {book_idx}")
            print(f"   user_num shape: {user_num.shape if user_num is not None else None}")
            print(f"   book_num shape: {book_num.shape if book_num is not None else None}")
            
            # Предсказание
            with torch.no_grad():
                score = self.model.forward_single(
                    torch.tensor([user_idx]), 
                    torch.tensor([book_idx]),
                    user_num, book_num
                )
            print(f"   ✅ Предсказание успешно: {score.item():.4f}")
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
        
        return vec
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
        self.user_cat_features = []
        self.book_cat_features = []
        
        if hasattr(self.trainer, 'user_num_cols'):
            self.user_num_features = self.trainer.user_num_cols
        elif hasattr(self.trainer, 'user_numerical_cols'):
            self.user_num_features = self.trainer.user_numerical_cols
        
        if hasattr(self.trainer, 'book_num_cols'):
            self.book_num_features = self.trainer.book_num_cols
        elif hasattr(self.trainer, 'book_numerical_cols'):
            self.book_num_features = self.trainer.book_numerical_cols
        
        if hasattr(self.trainer, 'user_cat_cols'):
            self.user_cat_features = self.trainer.user_cat_cols
        if hasattr(self.trainer, 'book_cat_cols'):
            self.book_cat_features = self.trainer.book_cat_cols
        
        if not self.user_num_features and hasattr(self.trainer, 'user_features') and self.trainer.user_features is not None:
            for col in self.trainer.user_features.columns:
                if self.trainer.user_features[col].dtype in ['float64', 'int64']:
                    self.user_num_features.append(col)
            logger.info(f"Автоматически определены user признаки: {self.user_num_features}")
        
        if not self.book_num_features and hasattr(self.trainer, 'book_features') and self.trainer.book_features is not None:
            for col in self.trainer.book_features.columns:
                if self.trainer.book_features[col].dtype in ['float64', 'int64']:
                    self.book_num_features.append(col)
            logger.info(f"Автоматически определены book признаки: {self.book_num_features}")
        
        self.feature_names = (
            [f'user_{col}' for col in self.user_num_features] +
            [f'book_{col}' for col in self.book_num_features] +
            [f'user_cat_{col}' for col in self.user_cat_features] +
            [f'book_cat_{col}' for col in self.book_cat_features]
        )
        
        logger.info(f"Определено {len(self.feature_names)} признаков для SHAP")
    
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
        
        vector = user_num + book_num
        
        if len(vector) == 0:
            print(f"   ❌ Пустой вектор для user={user_id}, book={book_id_str}")
            return None
        
        # print(f"   ✅ Вектор построен: {len(vector)} признаков")
        return np.array(vector, dtype=np.float32)

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
    
    def get_global_importance(self, test_pairs: List[Tuple[int, str]], 
                              nsamples_per_pair: int = 50,
                              max_pairs: int = 100,
                              save_plot: bool = True) -> Dict:
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
        
        mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
        mean_shap = shap_matrix.mean(axis=0)
        std_shap = shap_matrix.std(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
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
    
    def explain_prediction(self, user_id: int, book_id: str, nsamples: int = 100, save_plot: bool = True) -> Dict:
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
            
            explanation = {
                self.feature_names[i]: float(shap_1d[i]) 
                for i in range(len(self.feature_names))
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
            
            if 'error' not in explanation:
                results.append({
                    'book_id': book_id,
                    'score': score,
                    'top_positive': explanation['top_positive'][:3],
                    'top_negative': explanation['top_negative'][:3]
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
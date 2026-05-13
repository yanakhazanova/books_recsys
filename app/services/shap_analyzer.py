# app/services/shap_analyzer.py
import shap
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """Класс для SHAP интерпретации рекомендаций"""
    
    def __init__(self, model, trainer, device='cpu'):
        """
        Args:
            model: AFMRanker модель
            trainer: AFMTrainer с энкодерами и скейлерами
            device: 'cpu' или 'cuda'
        """
        self.model = model
        self.trainer = trainer
        self.device = device
        self.explainer = None
        self.background_data = None
        self.feature_names = None
        self.is_initialized = False
        
        # Определяем признаки
        self._define_features()
    
    def _define_features(self):
        """Определяет список всех признаков для SHAP"""
        # Безопасно получаем атрибуты
        self.user_num_features = []
        self.book_num_features = []
        self.user_cat_features = []
        self.book_cat_features = []
        
        # Пробуем разные возможные имена атрибутов
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
        
        # Если числовые признаки не найдены, пробуем определить из данных
        if not self.user_num_features and hasattr(self.trainer, 'user_features') and self.trainer.user_features is not None:
            # Берем все числовые колонки
            for col in self.trainer.user_features.columns:
                if self.trainer.user_features[col].dtype in ['float64', 'int64']:
                    self.user_num_features.append(col)
            print(f"Автоматически определены user числовые признаки: {self.user_num_features}")
        
        if not self.book_num_features and hasattr(self.trainer, 'book_features') and self.trainer.book_features is not None:
            for col in self.trainer.book_features.columns:
                if self.trainer.book_features[col].dtype in ['float64', 'int64']:
                    self.book_num_features.append(col)
            print(f"Автоматически определены book числовые признаки: {self.book_num_features}")
        
        self.feature_names = (
            [f'user_{col}' for col in self.user_num_features] +
            [f'book_{col}' for col in self.book_num_features] +
            [f'user_cat_{col}' for col in self.user_cat_features] +
            [f'book_cat_{col}' for col in self.book_cat_features]
        )
        
        print(f"Определено {len(self.feature_names)} признаков для SHAP")
        print(f"  User числовые: {len(self.user_num_features)}")
        print(f"  Book числовые: {len(self.book_num_features)}")
        
        if self.user_num_features:
            print(f"    Примеры user признаков: {self.user_num_features[:3]}")
        if self.book_num_features:
            print(f"    Примеры book признаков: {self.book_num_features[:3]}")
    
    def _build_feature_vector(self, user_id: int, book_id: str) -> Optional[np.ndarray]:
        """
        Строит вектор признаков для пары (user, book)
        
        Returns:
            np.ndarray: вектор признаков или None если данные не найдены
        """
        # Проверяем доступность данных
        if not hasattr(self.trainer, 'user_features') or self.trainer.user_features is None:
            print("user_features не загружены в trainer")
            return None
        
        if not hasattr(self.trainer, 'book_features') or self.trainer.book_features is None:
            print("book_features не загружены в trainer")
            return None
        
        # Проверяем существование пользователя
        if user_id not in self.trainer.user_features.index:
            logger.warning(f"Пользователь {user_id} не найден в user_features")
            return None
        
        # Проверяем существование книги (ISBN может быть в индексе или в колонке ISBN)
        book_id_str = str(book_id)
        
        # Пробуем найти книгу
        book_row = None
        if book_id_str in self.trainer.book_features.index:
            book_row = self.trainer.book_features.loc[book_id_str]
        elif 'ISBN' in self.trainer.book_features.columns:
            # Ищем по колонке ISBN
            mask = self.trainer.book_features['ISBN'].astype(str) == book_id_str
            if mask.any():
                book_row = self.trainer.book_features[mask].iloc[0]
            else:
                logger.warning(f"Книга {book_id_str} не найдена в book_features")
                return None
        else:
            logger.warning(f"Книга {book_id_str} не найдена в индексе book_features")
            return None
        
        user_f = self.trainer.user_features.loc[user_id]
        book_f = book_row
        
        # Числовые признаки пользователя
        user_num = []
        for col in self.user_num_features:
            if col in user_f.index:
                val = user_f[col]
                # Обработка NaN
                if pd.isna(val):
                    val = 0.0
                # Если это серия, берем первое значение
                if hasattr(val, 'iloc'):
                    val = val.iloc[0] if len(val) > 0 else 0.0
                user_num.append(float(val))
            else:
                user_num.append(0.0)
        
        # Числовые признаки книги
        book_num = []
        for col in self.book_num_features:
            if col in book_f.index:
                val = book_f[col]
                if pd.isna(val):
                    val = 0.0
                if hasattr(val, 'iloc'):
                    val = val.iloc[0] if len(val) > 0 else 0.0
                book_num.append(float(val))
            else:
                book_num.append(0.0)
        
        # Категориальные признаки (если есть)
        user_cat = []
        for col in self.user_cat_features:
            if col in user_f.index:
                val = user_f[col]
                if pd.isna(val):
                    val = 0
                user_cat.append(float(val) if not pd.isna(val) else 0.0)
            else:
                user_cat.append(0.0)
        
        book_cat = []
        for col in self.book_cat_features:
            if col in book_f.index:
                val = book_f[col]
                if pd.isna(val):
                    val = 0
                book_cat.append(float(val) if not pd.isna(val) else 0.0)
            else:
                book_cat.append(0.0)
        
        vector = user_num + book_num + user_cat + book_cat
        
        # Проверяем, что вектор не пустой
        if len(vector) == 0:
            logger.warning(f"Пустой вектор для user={user_id}, book={book_id_str}")
            return None
        
        return np.array(vector, dtype=np.float32)
    
    def _predict_fn(self, x: np.ndarray) -> np.ndarray:
        """
        Функция предсказания для SHAP
        
        Args:
            x: массив признаков [n_samples, n_features]
        
        Returns:
            np.ndarray: предсказанные скоры
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(x.shape[0]):
                try:
                    # Разделяем признаки
                    n_user_num = len(self.user_num_features)
                    n_book_num = len(self.book_num_features)
                    n_user_cat = len(self.user_cat_features)
                    
                    # Извлекаем части вектора
                    user_num_arr = x[i, :n_user_num] if n_user_num > 0 else np.array([])
                    book_num_arr = x[i, n_user_num:n_user_num + n_book_num] if n_book_num > 0 else np.array([])
                    
                    # Создаем тензоры
                    user_tensor = torch.tensor([0], device=self.device)
                    book_tensor = torch.tensor([0], device=self.device)
                    
                    user_num = torch.tensor(user_num_arr, dtype=torch.float32, device=self.device).unsqueeze(0) if len(user_num_arr) > 0 else None
                    book_num = torch.tensor(book_num_arr, dtype=torch.float32, device=self.device).unsqueeze(0) if len(book_num_arr) > 0 else None
                    
                    # Предсказание
                    score = self.model.forward_single(user_tensor, book_tensor, user_num, book_num)
                    predictions.append(score.item())
                    
                except Exception as e:
                    logger.warning(f"Ошибка предсказания для sample {i}: {e}")
                    predictions.append(0.0)
        
        return np.array(predictions)
    
    def initialize(self, background_pairs: List[Tuple[int, str]], n_samples: int = 100) -> Dict:
        """
        Инициализирует SHAP объяснитель на фоновых данных
        
        Args:
            background_pairs: список пар (user_id, book_id) для фона
            n_samples: количество сэмплов для SHAP
        
        Returns:
            Dict: статус инициализации
        """
        print(f"Инициализация SHAP на {len(background_pairs)} фоновых парах...")
        
        # Строим фоновую матрицу
        background_vectors = []
        valid_pairs = []
        
        for user_id, book_id in background_pairs:
            vec = self._build_feature_vector(user_id, book_id)
            if vec is not None:
                background_vectors.append(vec)
                valid_pairs.append((user_id, book_id))
            else:
                logger.debug(f"Не удалось построить вектор для user={user_id}, book={book_id}")
        
        print(f"Построено {len(background_vectors)} валидных векторов из {len(background_pairs)}")
        
        if len(background_vectors) < 10:
            # Выводим отладочную информацию
            print("Недостаточно валидных фоновых пар")
            print(f"trainer.user_features shape: {self.trainer.user_features.shape if hasattr(self.trainer, 'user_features') else 'None'}")
            print(f"trainer.book_features shape: {self.trainer.book_features.shape if hasattr(self.trainer, 'book_features') else 'None'}")
            
            # Показываем примеры пользователей и книг
            if hasattr(self.trainer, 'user_features') and self.trainer.user_features is not None:
                print(f"Примеры пользователей в user_features: {list(self.trainer.user_features.index[:5])}")
            if hasattr(self.trainer, 'book_features') and self.trainer.book_features is not None:
                print(f"Примеры книг в book_features: {list(self.trainer.book_features.index[:5])}")
            
            return {
                'status': 'error',
                'message': f'Недостаточно валидных фоновых пар: {len(background_vectors)} (нужно минимум 10)',
                'debug_info': {
                    'total_pairs': len(background_pairs),
                    'valid_vectors': len(background_vectors),
                    'user_features_shape': str(self.trainer.user_features.shape) if hasattr(self.trainer, 'user_features') else None,
                    'book_features_shape': str(self.trainer.book_features.shape) if hasattr(self.trainer, 'book_features') else None
                }
            }
        
        self.background_data = np.array(background_vectors)
        
        # Создаем объяснитель
        print("Создание KernelExplainer...")
        try:
            self.explainer = shap.KernelExplainer(self._predict_fn, self.background_data)
            self.is_initialized = True
        except Exception as e:
            print(f"Ошибка создания KernelExplainer: {e}")
            return {
                'status': 'error',
                'message': f'Ошибка создания объяснителя: {str(e)}'
            }
        
        return {
            'status': 'success',
            'message': f'SHAP инициализирован на {len(background_vectors)} фоновых парах',
            'n_background_samples': len(background_vectors),
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names[:10]  # Топ-10 для отладки
        }
    
    def explain_prediction(self, user_id: int, book_id: str, nsamples: int = 100) -> Optional[Dict]:
        """
        Объясняет предсказание для конкретной пары (user, book)
        """
        if not self.is_initialized:
            return {
                'error': 'SHAP не инициализирован. Сначала вызовите initialize()'
            }
        
        # Строим вектор признаков
        feature_vector = self._build_feature_vector(user_id, book_id)
        if feature_vector is None:
            return {
                'error': f'Не удалось построить вектор для user={user_id}, book={book_id}'
            }
        
        try:
            # Вычисляем SHAP значения
            shap_values = self.explainer.shap_values(feature_vector.reshape(1, -1), nsamples=nsamples)
            
            # Приводим к правильной размерности
            if len(shap_values.shape) == 3:
                shap_1d = shap_values[0, :, 0]
            elif len(shap_values.shape) == 2:
                shap_1d = shap_values[0, :]
            else:
                shap_1d = shap_values[0]
            
            # Создаем словарь признак -> SHAP значение
            explanation = {
                self.feature_names[i]: float(shap_1d[i]) 
                for i in range(len(self.feature_names))
            }
            
            # Добавляем предсказанный скор
            score = self._predict_fn(feature_vector.reshape(1, -1))[0]
            
            return {
                'user_id': user_id,
                'book_id': book_id,
                'predicted_score': float(score),
                'shap_values': explanation,
                'top_positive': sorted([(k, v) for k, v in explanation.items() if v > 0], 
                                      key=lambda x: x[1], reverse=True)[:10],
                'top_negative': sorted([(k, v) for k, v in explanation.items() if v < 0], 
                                      key=lambda x: x[1])[:10]
            }
        except Exception as e:
            print(f"Ошибка вычисления SHAP: {e}")
            return {
                'error': f'Ошибка: {str(e)}'
            }
    
    def get_global_importance(self, test_pairs: List[Tuple[int, str]], 
                              nsamples_per_pair: int = 50,
                              max_pairs: int = 100) -> pd.DataFrame:
        """Вычисляет глобальную важность признаков"""
        if not self.is_initialized:
            raise ValueError("SHAP не инициализирован. Сначала вызовите initialize()")
        
        # Ограничиваем количество пар
        test_pairs = test_pairs[:max_pairs]
        
        # Строим тестовую матрицу
        test_vectors = []
        valid_pairs = []
        
        for user_id, book_id in test_pairs:
            vec = self._build_feature_vector(user_id, book_id)
            if vec is not None:
                test_vectors.append(vec)
                valid_pairs.append((user_id, book_id))
        
        if len(test_vectors) == 0:
            return pd.DataFrame()
        
        X_test = np.array(test_vectors)
        
        # Вычисляем SHAP значения
        print(f"Вычисление SHAP для {len(test_vectors)} тестовых пар...")
        
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
                logger.warning(f"Ошибка для пары {i}: {e}")
                all_shap_values.append(np.zeros(len(self.feature_names)))
        
        if not all_shap_values:
            return pd.DataFrame()
        
        shap_matrix = np.array(all_shap_values)
        
        # Вычисляем среднюю абсолютную важность
        mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
        
        # Создаем DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_abs_shap,
            'mean_shap': shap_matrix.mean(axis=0),
            'std_shap': shap_matrix.std(axis=0)
        }).sort_values('mean_abs_shap', ascending=False)
        
        return importance_df
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Возвращает описания признаков"""
        descriptions = {}
        
        for col in self.user_num_features:
            descriptions[f'user_{col}'] = f'Признак пользователя: {col}'
        
        for col in self.book_num_features:
            descriptions[f'book_{col}'] = f'Признак книги: {col}'
        
        for col in self.user_cat_features:
            descriptions[f'user_cat_{col}'] = f'Категориальный признак пользователя: {col}'
        
        for col in self.book_cat_features:
            descriptions[f'book_cat_{col}'] = f'Категориальный признак книги: {col}'
        
        return descriptions
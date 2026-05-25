import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Dict, Optional, List
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle

from app.services.afm_model import AFMRanker, AFMDataset

import numpy as np
from app.core.config import settings

class TrainTestSplitter:
    """Разделение данных на тренировочные и тестовые"""
    
    def __init__(self, 
                 strong_pos_threshold: int = 8,
                 weak_pos_is_zero: bool = True,
                 neg_threshold: int = 5,
                 test_items_per_user: int = 2,
                 min_strong_pos: int = 3,
                 random_state: int = 42):
        
        self.strong_pos_threshold = strong_pos_threshold
        self.weak_pos_is_zero = weak_pos_is_zero
        self.neg_threshold = neg_threshold
        self.test_items_per_user = test_items_per_user
        self.min_strong_pos = min_strong_pos
        self.random_state = random_state
        
        np.random.seed(random_state)
    
    def _categorize_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Категоризация рейтингов"""
        df = df.copy()
        
        df.loc[df['Rating'] >= self.strong_pos_threshold, 'rating_category'] = 'strong_positive'
        
        if self.weak_pos_is_zero:
            df.loc[df['Rating'] == 0, 'rating_category'] = 'weak_positive'
        
        df.loc[(df['Rating'] > 0) & (df['Rating'] <= self.neg_threshold), 'rating_category'] = 'negative'
        
        return df
    
    def split_all_users(self, ratings_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split: ВСЕ пользователи в тесте, но с разными книгами
        Возвращает (train_df, test_df)
        """
        print("=" * 60)
        print("SPLIT: ВСЕ ПОЛЬЗОВАТЕЛИ В ТЕСТЕ")
        print("=" * 60)
        
        df = self._categorize_ratings(ratings_df)
        
        # Статистика strong_positive
        user_strong_counts = df[df['rating_category'] == 'strong_positive'].groupby('User-ID').size()
        
        print(f"\n📊 Статистика strong_positive:")
        print(f"   Пользователей с strong_positive: {len(user_strong_counts)}")
        print(f"   Среднее: {user_strong_counts.mean():.1f}")
        print(f"   Медиана: {user_strong_counts.median():.1f}")
        
        # Оставляем только пользователей с достаточным числом strong_positive
        qualified_users = user_strong_counts[user_strong_counts >= self.min_strong_pos].index
        print(f"\n✅ Пользователей для теста (≥{self.min_strong_pos} strong_pos): {len(qualified_users)}")
        
        # Формируем train/test
        train_rows = []
        test_rows = []
        
        for user in qualified_users:
            user_data = df[df['User-ID'] == user]
            strong_pos_data = user_data[user_data['rating_category'] == 'strong_positive']
            
            # Выбираем случайные strong_positive для теста
            n_test = min(self.test_items_per_user, len(strong_pos_data) - 1)
            test_indices = np.random.choice(strong_pos_data.index, size=n_test, replace=False)
            
            test_rows.append(user_data.loc[test_indices])
            train_rows.append(user_data.drop(test_indices))
        
        # Пользователи без strong_positive идут только в train
        other_users = df[~df['User-ID'].isin(qualified_users)]
        train_rows.append(other_users)
        
        # Собираем финальные датафреймы
        train_df = pd.concat(train_rows, ignore_index=True)
        test_df = pd.concat(test_rows, ignore_index=True)
        
        print(f"\n✅ Финальные размеры:")
        print(f"   Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
        print(f"   Test:  {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
        print(f"   Тестовых пользователей: {test_df['User-ID'].nunique()}")
        
        return train_df, test_df
    
    def split_random(self, ratings_df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Альтернативный вариант: случайное разбиение
        """
        from sklearn.model_selection import train_test_split
        
        users = ratings_df['User-ID'].unique()
        train_users, test_users = train_test_split(users, test_size=test_size, random_state=self.random_state)
        
        train_df = ratings_df[ratings_df['User-ID'].isin(train_users)]
        test_df = ratings_df[ratings_df['User-ID'].isin(test_users)]
        
        print(f"Random split: Train users={len(train_users)}, Test users={len(test_users)}")
        
        return train_df, test_df
    
from app.services.collab_filter import CollaborativeFilter, PopularityFallback, CandidateAnalyzer


class CollaborativeGenerator:
    """Генерация кандидатов коллаборативной фильтрацией"""
    
    def __init__(self, n_candidates: int = 1000):
        self.n_candidates = n_candidates
        self.cf = CollaborativeFilter()
        self.fallback = PopularityFallback()
        self.candidates = None
    
    def generate(self, train_ratings: pd.DataFrame, 
                 test_ratings: pd.DataFrame = None,
                 books_df: pd.DataFrame = None) -> Dict:
        """
        Полный pipeline генерации кандидатов
        """
        if books_df is None:
            raise ValueError("books_df обязателен для генерации кандидатов от авторов")
        
        # 1. Генерация кандидатов коллаборативной фильтрацией + авторские
        self.candidates = self.cf.generate_candidates(
            train_ratings, 
            books_df,
            n_recommendations=self.n_candidates
        )
        
        # 2. Добавляем популярные книги для малоактивных пользователей
        popular_books = self.fallback.get_top_popular_books(train_ratings, n=500)
        inactive_users = self.fallback.get_inactive_users(train_ratings, min_books=100)
        
        self.candidates = self.fallback.add_popular_to_inactive(
            recommendations=self.candidates, 
            inactive_users=inactive_users, 
            train_ratings=train_ratings,
            popular_books=popular_books,
            books_df=books_df,
            add_ratio=0.05
        )
        
        # 3. Анализ качества (если есть тестовые данные)
        if test_ratings is not None:
            from app.services.collab_filter import CandidateAnalyzer
            analyzer = CandidateAnalyzer()
            coverage = analyzer.analyze_coverage(self.candidates, test_ratings)
            print(f"\n📊 Покрытие тестовых пользователей: {coverage['coverage_rate']:.1%}")
            
            hit_rate = analyzer.calculate_hit_rate(self.candidates, test_ratings)
            print(f"   Hit rate: {hit_rate['mean_hit_rate']:.3f}")
        
        # 4. Сохраняем кандидатов
        self.cf.save_candidates()
        
        return self.candidates


import pandas as pd
import pickle
from pathlib import Path
from typing import Tuple, Dict
from app.core.config import settings
from app.services.feature_eng import BookFeatureEngineer, UserFeatureEngineer, TripletGenerator

class FeaturePipeline:
    """Pipeline для формирования фичей и триплетов"""
    
    def __init__(self):
        self.book_fe = BookFeatureEngineer(
            author_popularity_threshold=800,
            use_reads_based_popularity=True
        )
        self.user_fe = UserFeatureEngineer()
        self.triplet_gen = TripletGenerator()
        self.book_features = None
        self.user_features = None
        self.triplets = None
    
    def generate_features_and_triplets(self, 
                                       train_ratings: pd.DataFrame,
                                       books: pd.DataFrame,
                                       users: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Формирование фичей и триплетов только на тренировочных данных
        
        Args:
            train_ratings: тренировочные рейтинги (уже после split)
            books: данные о книгах
            users: данные о пользователях
            
        Returns:
            (book_features, user_features, triplets_df)
        """
        print("\n" + "=" * 60)
        print("ФОРМИРОВАНИЕ ФИЧЕЙ И ТРИПЛЕТОВ")
        print("=" * 60)
        
        # 1. Контентные фичи книг (без рейтингов)
        print("\n📚 Шаг 1: Контентные фичи книг...")
        book_content_features = self.book_fe.create_book_features_without_ratings(books, train_ratings)
        print(f"   Создано {len(book_content_features)} книг с {len(book_content_features.columns)} контентными фичами")
        
        # 2. Добавляем статистики из тренировочных рейтингов
        print("\n📊 Шаг 2: Добавление статистик из рейтингов...")
        self.book_features, _ = self.book_fe.add_rating_statistics(
            book_content_features, 
            train_ratings,  # Только тренировочные!
            fit_scaler=True
        )
        
        # 3. Фичи пользователей (только из тренировочных)
        print("\n👤 Шаг 3: Фичи пользователей...")
        self.user_features = self.user_fe.create_user_features(users, train_ratings)
        
        # 4. Триплеты (только из тренировочных)
        print("\n🔄 Шаг 4: Генерация триплетов...")
        self.triplets = TripletGenerator.generate_triplets(train_ratings)
        
        print("\n" + "=" * 60)
        print("✅ ФИЧИ И ТРИПЛЕТЫ ГОТОВЫ")
        print(f"   Размер book_features: {self.book_features.shape}")
        print(f"   Размер user_features: {self.user_features.shape}")
        print(f"   Количество триплетов: {len(self.triplets)}")
        print("=" * 60)
        
        print(f"\n📊 ПРОВЕРКА ФИЧЕЙ ПЕРЕД ОБУЧЕНИЕМ:")
        print(f"   User features shape: {self.user_features.shape}")
        print(f"   User features columns ({len(self.user_features.columns)}):")
        for i, col in enumerate(self.user_features.columns[:15]):
            print(f"      {i+1}. {col} (type: {self.user_features[col].dtype})")
        if len(self.user_features.columns) > 15:
            print(f"      ... и еще {len(self.user_features.columns)-15} колонок")
        
        print(f"\n   Book features shape: {self.book_features.shape}")
        print(f"   Book features columns ({len(self.book_features.columns)}):")
        for i, col in enumerate(self.book_features.columns[:15]):
            print(f"      {i+1}. {col} (type: {self.book_features[col].dtype})")
        if len(self.book_features.columns) > 15:
            print(f"      ... и еще {len(self.book_features.columns)-15} колонок")
        
        # Проверяем наличие one-hot колонок
        ohe_cols = [col for col in self.book_features.columns if any(x in col for x in ['Author_popular', 'Publisher_popular', 'Year_era'])]
        if ohe_cols:
            print(f"\n   ✅ Найдено one-hot колонок: {len(ohe_cols)}")
            print(f"      Примеры: {ohe_cols[:5]}")
        else:
            print(f"\n   ⚠️  ВНИМАНИЕ: Не найдено one-hot колонок! Author_popular не закодированы?")

        return self.book_features, self.user_features, self.triplets
    
    def save(self, filepath: str = "models/features_and_scalers.pkl"):
        """Сохраняет обученные трансформеры и фичи"""
        save_data = {
            'book_features': self.book_features,
            'user_features': self.user_features,
            'triplets': self.triplets,
            'book_scaler': self.book_fe.scaler,
            'book_ohe': self.book_fe.ohe,
            'user_scaler': self.user_fe.user_scaler,
            'user_ohe_activity': self.user_fe.ohe_activity,
            'user_ohe_age': self.user_fe.ohe_age,
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"💾 Фичи и скейлеры сохранены в {filepath}")
    
    def load(self, filepath: str = "models/features_and_scalers.pkl"):
        """Загружает обученные трансформеры и фичи"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.book_features = save_data['book_features']
        self.user_features = save_data['user_features']
        self.triplets = save_data['triplets']
        self.book_fe.scaler = save_data['book_scaler']
        self.book_fe.ohe = save_data['book_ohe']
        self.user_fe.user_scaler = save_data['user_scaler']
        self.user_fe.ohe_activity = save_data['user_ohe_activity']
        self.user_fe.ohe_age = save_data['user_ohe_age']
        
        # Отмечаем, что трансформеры обучены
        self.book_fe.fitted = True
        self.user_fe.fitted = True
        
        print(f"📂 Фичи и скейлеры загружены из {filepath}")
        return self.book_features, self.user_features, self.triplets


class AFMTrainer:
    """Обучалка AFM модели"""

    def __init__(self, 
                 embed_dim: int = 64,
                 attention_dim: int = 32,
                 dropout: float = 0.2,
                 batch_size: int = 512,
                 epochs: int = 10,
                 learning_rate: float = 0.001,
                 val_split: float = 0.2,
                 device: str = None):
        
        # Валидация параметров
        if embed_dim < 8 or embed_dim > 512:
            raise ValueError(f"embed_dim должен быть между 8 и 512, получено {embed_dim}")
        if attention_dim < 4 or attention_dim > 256:
            raise ValueError(f"attention_dim должен быть между 4 и 256, получено {attention_dim}")
        if dropout < 0 or dropout > 0.8:
            raise ValueError(f"dropout должен быть между 0 и 0.8, получено {dropout}")
        if batch_size < 1 or batch_size > 10000:
            raise ValueError(f"batch_size должен быть между 1 и 10000, получено {batch_size}")
        if epochs < 1 or epochs > 200:
            raise ValueError(f"epochs должен быть между 1 и 200, получено {epochs}")
        if learning_rate <= 0 or learning_rate > 1:
            raise ValueError(f"learning_rate должен быть между 0 и 1, получено {learning_rate}")
        if val_split < 0 or val_split >= 1:
            raise ValueError(f"val_split должен быть между 0 и 1, получено {val_split}")
        
        self.embed_dim = embed_dim
        self.attention_dim = attention_dim
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.val_split = val_split
        
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.train_losses = []
        self.val_losses = []
        
        # Атрибуты для хранения данных (будут заполнены при fit или load)
        self.user_encoder = None
        self.book_encoder = None
        self.user_scaler = None
        self.book_scaler = None
        self.user_cat_cols = []
        self.book_cat_cols = []
        self.user_numerical_cols = []
        self.book_numerical_cols = []
        self.user_features = None  # Будет установлен при предсказаниях
        self.book_features = None  # Будет установлен при предсказаниях
        self.user_clean_to_original = {}
        self.book_clean_to_original = {}
        
        print(f"✅ AFMTrainer инициализирован с параметрами:")
        print(f"   embed_dim={embed_dim}, attention_dim={attention_dim}")
        print(f"   batch_size={batch_size}, epochs={epochs}, lr={learning_rate}")
        print(f"   device={self.device}")
    
    def _get_categorical_value(self, features_df, col_name, default=0):
        """
        Безопасно получает значение категориальной фичи из DataFrame
        """
        if col_name not in features_df.index:
            return default
        
        val = features_df[col_name]
        
        # Если это Series с несколькими значениями, берем первое
        if hasattr(val, 'iloc'):
            if len(val) > 0:
                val = val.iloc[0]
            else:
                return default
        
        # Проверяем на NaN
        if pd.isna(val):
            return default
        
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    def prepare_dataloaders(self, triplets_df: pd.DataFrame, 
                        user_features: pd.DataFrame, 
                        book_features: pd.DataFrame) -> Tuple[DataLoader, DataLoader, AFMDataset]:
        """
        Подготовка dataloaders для обучения
        """
        print("\n📊 Подготовка датасета...")
        
        print(f"\n🔍 ВХОДНЫЕ ДАННЫЕ ДЛЯ DATASET:")
        print(f"   user_features shape: {user_features.shape}")
        print(f"   book_features shape: {book_features.shape}")
        print(f"   triplets count: {len(triplets_df)}")

        # Создаем датасет с fit_encoders=True
        dataset = AFMDataset(
            triplets_df=triplets_df,
            user_features=user_features,
            book_features=book_features,
            fit_encoders=True  # Важно!
        )
        
        print(f"\n✅ ДАТАСЕТ СОЗДАН:")
        print(f"   User числовых колонок: {len(dataset.user_num_cols)}")
        print(f"   Book числовых колонок: {len(dataset.book_num_cols)}")
        print(f"   Всего признаков на входе: {len(dataset.user_num_cols) + len(dataset.book_num_cols)}")
        
        if hasattr(dataset, 'user_num_cols') and dataset.user_num_cols:
            print(f"   Примеры user_num_cols: {dataset.user_num_cols[:5]}")
        if hasattr(dataset, 'book_num_cols') and dataset.book_num_cols:
            print(f"   Примеры book_num_cols: {dataset.book_num_cols[:5]}")

        # Разделяем на train/val
        train_size = int((1 - self.val_split) * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Создаем DataLoaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=0
        )
        
        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Val samples: {len(val_dataset)}")
        print(f"   User numerical features: {len(dataset.user_num_cols)}")
        print(f"   Book numerical features: {len(dataset.book_num_cols)}")
        
        # Сохраняем encoder'ы и scaler'ы из dataset
        self.user_encoder = dataset.user_encoder
        self.book_encoder = dataset.book_encoder
        self.user_scaler = dataset.user_scaler
        self.book_scaler = dataset.book_scaler
        self.user_num_cols = dataset.user_num_cols
        self.book_num_cols = dataset.book_num_cols
        
        return train_loader, val_loader, dataset

    def create_model(self, dataset: AFMDataset) -> AFMRanker:
        """
        Создает модель с правильными размерами
        """

        print(f"\n🏗️  СОЗДАНИЕ МОДЕЛИ:")
        print(f"   user_count: {len(dataset.user_encoder.classes_)}")
        print(f"   book_count: {len(dataset.book_encoder.classes_)}")
        print(f"   user_num_dim: {len(dataset.user_num_cols)}")
        print(f"   book_num_dim: {len(dataset.book_num_cols)}")
        print(f"   Всего числовых признаков: {len(dataset.user_num_cols) + len(dataset.book_num_cols)}")
        
        if len(dataset.user_num_cols) < 10:
            print(f"   ⚠️  ВНИМАНИЕ: Мало user признаков ({len(dataset.user_num_cols)}). Ожидалось больше!")
        if len(dataset.book_num_cols) < 50:
            print(f"   ⚠️  ВНИМАНИЕ: Мало book признаков ({len(dataset.book_num_cols)}). Ожидалось больше (есть one-hot?).")

        model = AFMRanker(
            user_count=len(dataset.user_encoder.classes_),
            book_count=len(dataset.book_encoder.classes_),
            user_num_dim=len(dataset.user_num_cols),  # Изменено с user_numerical_cols
            book_num_dim=len(dataset.book_num_cols),  # Изменено с book_numerical_cols
            embed_dim=self.embed_dim,
            attention_dim=self.attention_dim,
            dropout=self.dropout
        ).to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n🤖 Модель создана:")
        print(f"   Параметров: {total_params:,}")
        print(f"   User эмбеддингов: {len(dataset.user_encoder.classes_)}")
        print(f"   Book эмбеддингов: {len(dataset.book_encoder.classes_)}")
        print(f"   User числовых фичей: {len(dataset.user_num_cols)}")
        print(f"   Book числовых фичей: {len(dataset.book_num_cols)}")
        
        return model

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Tuple[list, list]:
        """
        Обучение модели
        """
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
        
        self.train_losses = []
        self.val_losses = []
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            total_train_loss = 0
            
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
            for batch in train_bar:
                # Переносим на устройство
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                optimizer.zero_grad()
                loss, _, _ = self.model(batch)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                train_bar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = total_train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            
            # Validation
            self.model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    loss, _, _ = self.model(batch)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            self.val_losses.append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        return self.train_losses, self.val_losses


    def continue_training(self, 
                        triplets_df: pd.DataFrame,
                        user_features: pd.DataFrame,
                        book_features: pd.DataFrame,
                        additional_epochs: int = 5,
                        learning_rate: Optional[float] = None,
                        reset_optimizer: bool = False) -> 'AFMTrainer':
        """
        Продолжает обучение уже существующей модели
        
        Args:
            triplets_df: данные для дообучения
            user_features: фичи пользователей
            book_features: фичи книг
            additional_epochs: сколько дополнительных эпох обучить
            learning_rate: новая learning rate (если None, использует текущую * 0.5)
            reset_optimizer: сбросить ли оптимизатор (если меняем данные)
        
        Returns:
            self
        """
        if self.model is None:
            raise ValueError("Модель не загружена. Сначала вызовите load() или fit()")
        
        print("\n" + "="*60)
        print(f"🔄 ПРОДОЛЖЕНИЕ ОБУЧЕНИЯ (+{additional_epochs} эпох)")
        print("="*60)
        
        original_epochs = self.epochs
        self.epochs += additional_epochs
        
        # Настраиваем learning rate
        if learning_rate is None:
            learning_rate = self.learning_rate * 0.5  # Уменьшаем LR для тонкой настройки
            print(f"   Learning rate уменьшена: {self.learning_rate} → {learning_rate}")
        
        self.learning_rate = learning_rate
        
        # Создаем оптимизатор заново если нужно
        if reset_optimizer:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            print("   Оптимизатор сброшен")
        
        # Подготовка данных (может быть только часть данных для дообучения)
        train_loader, val_loader, _ = self.prepare_dataloaders(
            triplets_df, user_features, book_features
        )
        
        # Создаем scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=2, factor=0.5
        )
        
        # Дополнительное обучение
        start_epoch = len(self.train_losses)
        
        for epoch in range(additional_epochs):
            # Training
            self.model.train()
            total_train_loss = 0
            
            train_bar = tqdm(train_loader, desc=f"Fine-tune Epoch {start_epoch+epoch+1}/{self.epochs} [Train]")
            for batch in train_bar:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                self.optimizer.zero_grad()
                loss, _, _ = self.model(batch)
                loss.backward()
                
                # Gradient clipping для стабильности
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_train_loss += loss.item()
                train_bar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = total_train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            
            # Validation
            self.model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    loss, _, _ = self.model(batch)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            self.val_losses.append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            
            print(f"Epoch {start_epoch+epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        print("\n✅ Дообучение завершено!")
        print(f"   Всего эпох: {self.epochs}")
        print(f"   Best val loss: {min(self.val_losses):.4f}")
        
        return self


    def set_optimizer(self):
        """Создает или обновляет оптимизатор"""
        if not hasattr(self, 'optimizer') or self.optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)



    def fit(self, triplets_df: pd.DataFrame, 
            user_features: pd.DataFrame, 
            book_features: pd.DataFrame) -> 'AFMTrainer':
        """
        Основной метод для обучения модели
        """
        # Сохраняем ссылки на данные для предсказаний
        self.user_features = user_features
        self.book_features = book_features
        
        # Подготовка данных
        train_loader, val_loader, dataset = self.prepare_dataloaders(
            triplets_df, user_features, book_features
        )
        
        # Создание модели
        self.model = self.create_model(dataset)
        
        # Создаем оптимизатор
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Обучение
        print(f"\n🚀 Начинаем обучение на {self.epochs} эпох...")
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        
        self.train_losses = []
        self.val_losses = []
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            total_train_loss = 0
            
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
            for batch in train_bar:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                self.optimizer.zero_grad()
                loss, _, _ = self.model(batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_train_loss += loss.item()
                train_bar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = total_train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            
            # Validation
            self.model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    loss, _, _ = self.model(batch)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            self.val_losses.append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Сохраняем encoder'ы и scaler'ы
        self.user_encoder = dataset.user_encoder
        self.book_encoder = dataset.book_encoder
        self.user_scaler = dataset.user_scaler
        self.book_scaler = dataset.book_scaler
        self.user_num_cols = dataset.user_num_cols
        self.book_num_cols = dataset.book_num_cols
        
        return self

    def save(self, filepath: str = "models/afm_model.pth"):
        """Сохраняет модель и все необходимые компоненты"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем всё, что нужно для восстановления
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
            'user_encoder': self.user_encoder,
            'book_encoder': self.book_encoder,
            'user_scaler': self.user_scaler,
            'book_scaler': self.book_scaler,
            'user_num_cols': self.user_num_cols,
            'book_num_cols': self.book_num_cols,
            'embed_dim': self.embed_dim,
            'attention_dim': self.attention_dim,
            'dropout': self.dropout,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
        }, filepath)
        
        print(f"💾 Модель сохранена в {filepath}")
        print(f"   Обучено эпох: {len(self.train_losses)}")
        print(f"   Финальный train loss: {self.train_losses[-1]:.4f}")
        print(f"   Финальный val loss: {self.val_losses[-1]:.4f}")

    
    def load(self, filepath: str = "models/afm_model.pth"):
        """Загружает модель и все компоненты"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Модель не найдена: {filepath}")
        
        # Загружаем checkpoint
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # Восстанавливаем параметры
        self.embed_dim = checkpoint['embed_dim']
        self.attention_dim = checkpoint['attention_dim']
        self.dropout = checkpoint.get('dropout', 0.2)
        self.epochs = checkpoint.get('epochs', self.epochs)
        self.learning_rate = checkpoint.get('learning_rate', self.learning_rate)
        
        self.user_encoder = checkpoint['user_encoder']
        self.book_encoder = checkpoint['book_encoder']
        self.user_scaler = checkpoint['user_scaler']
        self.book_scaler = checkpoint['book_scaler']
        self.user_num_cols = checkpoint.get('user_num_cols', [])
        self.book_num_cols = checkpoint.get('book_num_cols', [])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        # Создаем модель
        self.model = AFMRanker(
            user_count=len(self.user_encoder.classes_),
            book_count=len(self.book_encoder.classes_),
            user_num_dim=len(self.user_num_cols),
            book_num_dim=len(self.book_num_cols),
            embed_dim=self.embed_dim,
            attention_dim=self.attention_dim,
            dropout=self.dropout
        ).to(self.device)
        
        # Загружаем веса
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Восстанавливаем оптимизатор
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if checkpoint.get('optimizer_state_dict') is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("   Оптимизатор восстановлен")
        
        self.user_features = None
        self.book_features = None
        
        print(f"📂 Модель загружена из {filepath}")
        print(f"   Всего эпох: {len(self.train_losses)}")
        print(f"   Финальный loss: {self.train_losses[-1] if self.train_losses else 'N/A':.4f}")
        
        return self


    def predict_for_user(self, user_id: int, book_ids: List[str], top_n: int = None) -> List[Tuple[str, float]]:
        """Предсказание для одного пользователя и списка книг"""
        if self.model is None:
            raise ValueError("Модель не загружена")
        
        if self.user_features is None or self.book_features is None:
            raise ValueError("Данные не загружены")
        
        self.model.eval()
        
        # Проверяем, известен ли пользователь
        if user_id not in self.user_encoder.classes_:
            print(f"⚠️ Пользователь {user_id} не известен модели")
            return []
        
        user_idx = self.user_encoder.transform([user_id])[0]
        
        # Числовые фичи пользователя
        if user_id in self.user_features.index:
            user_f = self.user_features.loc[user_id]
            user_num_arr = self.user_scaler.transform(
                [user_f[self.user_num_cols].fillna(0).values]
            )[0]
        else:
            user_num_arr = np.zeros(len(self.user_num_cols))
        
        user_num = torch.tensor(user_num_arr, dtype=torch.float32, device=self.device)
        
        scores = []
        valid_books = []
        
        for book_id in book_ids:
            if book_id not in self.book_encoder.classes_:
                continue
            
            book_idx = self.book_encoder.transform([book_id])[0]
            
            # Числовые фичи книги
            if book_id in self.book_features.index:
                book_f = self.book_features.loc[book_id]
                book_num_arr = self.book_scaler.transform(
                    [book_f[self.book_num_cols].fillna(0).values]
                )[0]
            else:
                book_num_arr = np.zeros(len(self.book_num_cols))
            
            book_num = torch.tensor(book_num_arr, dtype=torch.float32, device=self.device)
            
            with torch.no_grad():
                # Используем forward_single для предсказания
                score = self.model.forward_single(
                    user_idx, book_idx,
                    user_num.unsqueeze(0), book_num.unsqueeze(0)
                )
                scores.append(score.item())
                valid_books.append(book_id)
        
        # Сортируем по убыванию
        results = sorted(zip(valid_books, scores), key=lambda x: x[1], reverse=True)
        
        if top_n:
            results = results[:top_n]
        
        return results

    def predict_for_all_users(self, candidates_dict: Dict[int, Dict[str, float]], 
                            top_n: int = 10,
                            sample_users: int = None) -> Dict[int, List[str]]:
        """
        Предсказание для всех пользователей
        
        Args:
            candidates_dict: {user_id: {isbn: score}} - кандидаты от CF
            top_n: количество рекомендаций на пользователя
            sample_users: если указан, предсказать только для N случайных пользователей
        """
        if self.model is None:
            raise ValueError("Модель не загружена")
        
        self.model.eval()
        
        # Ограничиваем выборку если нужно
        if sample_users and sample_users < len(candidates_dict):
            import random
            user_ids = list(candidates_dict.keys())
            sampled_user_ids = random.sample(user_ids, sample_users)
            filtered_candidates = {user: candidates_dict[user] for user in sampled_user_ids}
            print(f"📊 Предсказания для выборки {len(filtered_candidates)} пользователей (из {len(candidates_dict)})")
        else:
            filtered_candidates = candidates_dict
            print(f"📊 Предсказания для всех {len(filtered_candidates)} пользователей")
        
        recommendations = {}
        
        # Используем tqdm для прогресс-бара
        from tqdm import tqdm
        for user_id, candidate_books in tqdm(filtered_candidates.items(), desc="Ранжирование"):
            if user_id not in self.user_encoder.classes_:
                recommendations[user_id] = list(candidate_books.keys())[:top_n]
                continue
            
            ranked = self.predict_for_user(user_id, list(candidate_books.keys()), top_n=top_n)
            recommendations[user_id] = [book_id for book_id, _ in ranked]
        
        return recommendations

    

import pickle
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd

class DataCache:
    """Кэш для хранения промежуточных данных между эндпоинтами"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cleaned_data_path = self.cache_dir / "cleaned_data.pkl"
        self.split_path = self.cache_dir / "split.pkl"
        self.candidates_path = self.cache_dir / "candidates.pkl"
        self.features_path = self.cache_dir / "features.pkl"
        self.model_path = self.cache_dir / "model.pkl"
    
    def save_cleaned_data(self, ratings: pd.DataFrame, books: pd.DataFrame, users: pd.DataFrame):
        """Сохраняет очищенные данные"""
        with open(self.cleaned_data_path, 'wb') as f:
            pickle.dump({'ratings': ratings, 'books': books, 'users': users}, f)
        print(f"💾 Очищенные данные сохранены в {self.cleaned_data_path}")
    
    def load_cleaned_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Загружает очищенные данные"""
        if not self.cleaned_data_path.exists():
            return None, None, None
        with open(self.cleaned_data_path, 'rb') as f:
            data = pickle.load(f)
        return data['ratings'], data['books'], data['users']
    
    def check_cleaned_data(self) -> bool:
        return self.cleaned_data_path.exists()
    
    def save_split(self, train_ratings: pd.DataFrame, test_ratings: pd.DataFrame):
        """Сохраняет разделенные данные"""
        with open(self.split_path, 'wb') as f:
            pickle.dump({'train': train_ratings, 'test': test_ratings}, f)
        print(f"💾 Сплиты сохранены в {self.split_path}")
    
    def load_split(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Загружает разделенные данные"""
        if not self.split_path.exists():
            return None, None
        with open(self.split_path, 'rb') as f:
            data = pickle.load(f)
        return data['train'], data['test']
    
    def check_split(self) -> bool:
        return self.split_path.exists()
    
    def save_candidates(self, candidates: dict):
        """Сохраняет кандидатов"""
        with open(self.candidates_path, 'wb') as f:
            pickle.dump(candidates, f)
        print(f"💾 Кандидаты сохранены в {self.candidates_path}")
    
    def load_candidates(self) -> Optional[dict]:
        """Загружает кандидатов"""
        if not self.candidates_path.exists():
            return None
        with open(self.candidates_path, 'rb') as f:
            return pickle.load(f)
    
    def check_candidates(self) -> bool:
        return self.candidates_path.exists()
    
    def save_features(self, book_features: pd.DataFrame, user_features: pd.DataFrame, triplets: pd.DataFrame):
        """Сохраняет фичи и триплеты"""
        with open(self.features_path, 'wb') as f:
            pickle.dump({'book_features': book_features, 'user_features': user_features, 'triplets': triplets}, f)
        print(f"💾 Фичи сохранены в {self.features_path}")
    
    def load_features(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Загружает фичи и триплеты"""
        if not self.features_path.exists():
            return None, None, None
        with open(self.features_path, 'rb') as f:
            data = pickle.load(f)
        return data['book_features'], data['user_features'], data['triplets']
    
    def check_features(self) -> bool:
        return self.features_path.exists()
    
    def save_model(self, model):
        """Сохраняет модель"""
        with open(self.model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"💾 Модель сохранена в {self.model_path}")
    
    def load_model(self):
        """Загружает модель"""
        if not self.model_path.exists():
            return None
        with open(self.model_path, 'rb') as f:
            return pickle.load(f)
    
    def check_model(self) -> bool:
        return self.model_path.exists()
    
    def clear(self):
        """Очищает весь кэш"""
        for path in [self.cleaned_data_path, self.split_path, self.candidates_path, 
                     self.features_path, self.model_path]:
            if path.exists():
                path.unlink()
        print("🧹 Кэш очищен")



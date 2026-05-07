# app/services/feature_eng.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy import stats
from typing import Tuple, Dict
import pickle
import joblib

class BookFeatureEngineer:
    """Инжиниринг фичей для книг (только контентные фичи, без утечек)"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.ohe = None
        self.fitted = False
    
    def create_content_features(self, books: pd.DataFrame) -> pd.DataFrame:
        """
        Создает контентные фичи книг (не использует рейтинги)
        Нет утечек данных!
        """
        books = books.copy()
        
        # Очистка названия
        books['Title'] = books['Title'].str.replace(r'\(.*?\)', '', regex=True)
        books['Title'] = books['Title'].str.replace(r'[^\w\s]', '', regex=True)
        books['Title'] = books['Title'].str.lower()
        
        # Популярность автора (только на основе самих книг, не рейтингов)
        books['Author_mentioned'] = books.groupby('Author')['ISBN'].transform('count')
        books['is_popular_author'] = (books['Author_mentioned'] > 10).astype(int)
        
        books['Author_popular'] = np.where(
            books['is_popular_author'] == 1, 
            books['Author'], 
            'unpopular'
        )
        
        # Популярность издателя (только на основе самих книг)
        books['Publisher_popular'] = np.where(
            books.groupby('Publisher')['Publisher'].transform('count') > 100,
            books['Publisher'],
            'unpopular'
        )
        
        # Возраст книги
        current_year = 2026
        books['book_age'] = current_year - books['Year']
        books['is_classic'] = (books['book_age'] > 50).astype(int)
        
        # Количество книг у издателя
        publisher_counts = books['Publisher'].value_counts()
        books['publisher_book_count'] = books['Publisher'].map(publisher_counts)
        
        # Временные эпохи
        bins = list(range(1900, 2021, 20))
        labels = [f'{bins[i]}-{bins[i+1]-1}' if bins[i+1] != float('inf') else f'{bins[i]}+' 
                  for i in range(len(bins)-1)]
        labels[0] = '<1900'
        books['Year_era'] = pd.cut(books['Year'], bins=bins, labels=labels, right=False)
        
        return books
    
    def create_book_features_without_ratings(self, books: pd.DataFrame) -> pd.DataFrame:
        """
        Создает фичи книг только на основе контента (без статистик из рейтингов)
        """
        books = self.create_content_features(books)
        
        # Числовые фичи
        numeric_features = books[['ISBN', 'Year', 'publisher_book_count', 'book_age', 'is_classic']].copy()
        
        if 'Author_mentioned' in books.columns:
            numeric_features['Author_mentioned'] = books['Author_mentioned']
        if 'is_popular_author' in books.columns:
            numeric_features['is_popular_author'] = books['is_popular_author']
        
        # One-Hot Encoding для категориальных
        categorical_cols = ['Author_popular', 'Publisher_popular', 'Year_era']
        categorical_cols = [col for col in categorical_cols if col in books.columns]
        
        if categorical_cols:
            self.ohe = OneHotEncoder(sparse_output=False, drop='first')
            encoded_features = self.ohe.fit_transform(books[categorical_cols])

            clean_columns = [clean_column_name(col) for col in self.ohe.get_feature_names_out(categorical_cols)]
            
            encoded_df = pd.DataFrame(
                encoded_features,
                columns=clean_columns,
                index=books.index
            )
            
            book_features = pd.concat([numeric_features, encoded_df], axis=1)
        else:
            book_features = numeric_features
        
        book_features = book_features.set_index('ISBN')
        
        return book_features
    
    def add_rating_statistics(self, 
                             book_features: pd.DataFrame, 
                             train_ratings: pd.DataFrame,
                             fit_scaler: bool = True) -> Tuple[pd.DataFrame, StandardScaler]:
        """
        ДОБАВЛЯЕТ статистики из рейтингов (ТОЛЬКО на тренировочных данных!)
        Теперь утечки нет, потому что train_ratings - это ИСТИННО тренировочные данные
        """
        # Статистики ТОЛЬКО из тренировочных данных
        rating_stats = train_ratings.groupby('ISBN').agg({
            'Rating': ['count', 'mean', 'std', 'min', 'max'],
            'User-ID': 'nunique'
        }).reset_index()
        
        rating_stats.columns = ['ISBN', 'book_rating_count', 'book_avg_rating',
                                'book_rating_std', 'book_min_rating', 'book_max_rating',
                                'unique_users_rated']
        
        # Wilson score (только на тренировочных)
        def wilson_score(avg_rating, n_ratings, confidence=0.95):
            if n_ratings == 0:
                return 0
            p = avg_rating / 10.0
            z = stats.norm.ppf(1 - (1 - confidence) / 2)
            return (p + z*z/(2*n_ratings) - z * np.sqrt((p*(1-p) + z*z/(4*n_ratings))/n_ratings)) / (1 + z*z/n_ratings)
        
        rating_stats['wilson_score'] = rating_stats.apply(
            lambda x: wilson_score(x['book_avg_rating'], x['book_rating_count']), 
            axis=1
        )
        
        rating_stats['popularity_norm'] = rating_stats['book_rating_count'] / rating_stats['book_rating_count'].max()
        
        # Объединяем
        book_features = book_features.merge(rating_stats, on='ISBN', how='left')
        
        # Заполняем пропуски для новых книг
        default_values = {
            'book_rating_count': 0,
            'book_avg_rating': 0,
            'book_rating_std': 0,
            'book_min_rating': 0,
            'book_max_rating': 0,
            'unique_users_rated': 0,
            'wilson_score': 0,
            'popularity_norm': 0
        }
        book_features = book_features.fillna(default_values)
        
        # Нормализация числовых фичей (только на тренировочных данных)
        numerical_cols = ['book_rating_count', 'book_avg_rating', 'book_rating_std', 
                          'book_min_rating', 'book_max_rating', 'unique_users_rated',
                          'wilson_score', 'popularity_norm', 'book_age', 'publisher_book_count']
        
        # Проверяем, какие колонки реально существуют
        numerical_cols = [col for col in numerical_cols if col in book_features.columns]
        
        if fit_scaler:
            # Обучаем scaler на тренировочных данных
            book_features[numerical_cols] = self.scaler.fit_transform(
                book_features[numerical_cols].fillna(0)
            )
            self.fitted = True
        else:
            # Применяем уже обученный scaler
            if not self.fitted:
                raise ValueError("Scaler не обучен. Сначала вызовите с fit_scaler=True")
            book_features[numerical_cols] = self.scaler.transform(
                book_features[numerical_cols].fillna(0)
            )
        
        # Удаляем дубликаты индексов
        book_features = book_features.loc[~book_features.index.duplicated(keep='first')]
        
        print(f"✅ Создано {len(book_features)} книг с {len(book_features.columns)} фичами")
        
        return book_features, self.scaler


class UserFeatureEngineer:
    """Инжиниринг фичей для пользователей (без утечек)"""
    
    def __init__(self):
        self.user_scaler = StandardScaler()
        self.ohe_activity = None
        self.ohe_age = None
        self.fitted = False
    
    def create_user_features(self, users: pd.DataFrame, train_ratings: pd.DataFrame) -> pd.DataFrame:
        """
        Создает фичи пользователей на основе ТОЛЬКО тренировочных рейтингов
        """
        # Возрастные категории
        users = users.copy()
        users['is_age_known'] = users['Age'].apply(lambda x: x if pd.notnull(x) else 'unknown')
        
        bins = [0, 10, 20, 30, 40, 50, 60, 70, float('inf')]
        labels = ['<10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '>70']
        
        users['age_category'] = pd.cut(users['Age'], bins=bins, labels=labels, right=False)
        users['age_category'] = users['age_category'].cat.add_categories('unknown').fillna('unknown')
        
        # One-Hot для возраста
        self.ohe_age = OneHotEncoder(sparse_output=False, drop='first')
        age_encoded = self.ohe_age.fit_transform(users[['age_category']])
        age_df = pd.DataFrame(
            age_encoded,
            columns=self.ohe_age.get_feature_names_out(['age_category']),
            index=users.index
        )
        
        # Статистики из тренировочных рейтингов
        def calculate_user_features(group):
            non_zero_ratings = group[group['Rating'] > 0]['Rating']
            
            avg_rating = non_zero_ratings.mean() if len(non_zero_ratings) > 0 else 0
            min_rating = non_zero_ratings.min() if len(non_zero_ratings) > 0 else 0
            rating_std = non_zero_ratings.std() if len(non_zero_ratings) > 0 else 0
            
            non_zero_count = len(non_zero_ratings)
            zero_count = len(group[group['Rating'] == 0])
            total_count = len(group)
            
            non_zero_ratio = non_zero_count / total_count if total_count > 0 else 0
            
            return pd.Series({
                'user_rating_count': total_count,
                'non_zero_ratings_count': non_zero_count,
                'zero_ratings_count': zero_count,
                'non_zero_ratio': non_zero_ratio,
                'user_avg_rating': avg_rating,
                'user_min_rating': min_rating,
                'user_rating_std': rating_std,
                'user_max_rating': group['Rating'].max(),
                'unique_books_rated': group['ISBN'].nunique()
            })
        
        user_stats = train_ratings.groupby('User-ID').apply(calculate_user_features).reset_index()
        user_stats['user_rating_range'] = user_stats['user_max_rating'] - user_stats['user_min_rating']
        user_stats['user_rating_variability'] = user_stats['user_rating_std'].fillna(0)
        
        # Активность пользователя
        user_stats['user_activity_level'] = pd.cut(
            user_stats['user_rating_count'],
            bins=[0, 5, 20, 100, float('inf')],
            labels=['inactive', 'casual', 'active', 'super_active']
        )
        
        self.ohe_activity = OneHotEncoder(sparse_output=False, drop='first')
        activity_encoded = self.ohe_activity.fit_transform(user_stats[['user_activity_level']])
        activity_df = pd.DataFrame(
            activity_encoded,
            columns=self.ohe_activity.get_feature_names_out(['user_activity_level']),
            index=user_stats.index
        )
        
        # Объединяем все фичи
        user_features = user_stats.drop('user_activity_level', axis=1)
        user_features = pd.concat([user_features, activity_df, age_df], axis=1)
        user_features = user_features.set_index('User-ID')
        
        # Переименовываем колонки
        user_features.columns = user_features.columns.str.replace('user_activity_level_', 'activity_')
        user_features.columns = user_features.columns.str.replace('age_category_', 'age_')
        
        # Нормализуем числовые фичи
        numerical_cols = ['user_rating_count', 'non_zero_ratings_count', 'zero_ratings_count',
                         'non_zero_ratio', 'user_avg_rating', 'user_rating_std',
                         'user_rating_range', 'user_rating_variability']
        
        numerical_cols = [col for col in numerical_cols if col in user_features.columns]
        
        user_features[numerical_cols] = self.user_scaler.fit_transform(
            user_features[numerical_cols].fillna(0)
        )
        self.fitted = True
        
        print(f"✅ Создано {len(user_features)} пользователей с {len(user_features.columns)} фичами")
        
        return user_features
    
    def transform_user_features(self, users: pd.DataFrame, train_ratings: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет уже обученные преобразования к новым пользователям
        """
        if not self.fitted:
            raise ValueError("Feature engineer не обучен. Сначала вызовите create_user_features")
        
        # Аналогичная логика, но используем обученные scaler и encoders
        # (для production использования)
        pass


class TripletGenerator:
    """Генерация триплетов для pairwise ранжирования (только на тренировочных данных)"""
    
    @staticmethod
    def generate_triplets(train_ratings: pd.DataFrame,
                         strong_pos_threshold: int = 8,
                         weak_pos_threshold: int = 0,
                         neg_threshold: int = 5,
                         samples_per_user: int = 50) -> pd.DataFrame:
        """
        Генерирует триплеты ТОЛЬКО из тренировочных данных
        Нет утечек тестовых данных!
        """
        def get_category(rating):
            if rating >= strong_pos_threshold:
                return 'strong'
            elif rating == weak_pos_threshold:
                return 'weak'
            elif rating <= neg_threshold and rating > 0:
                return 'neg'
            else:
                return 'neutral'
        
        train_ratings = train_ratings.copy()
        train_ratings['category'] = train_ratings['Rating'].apply(get_category)
        
        triplets = []
        
        for user_id, user_data in train_ratings.groupby('User-ID'):
            strong_books = user_data[user_data['category'] == 'strong']['ISBN'].tolist()
            weak_books = user_data[user_data['category'] == 'weak']['ISBN'].tolist()
            neg_books = user_data[user_data['category'] == 'neg']['ISBN'].tolist()
            
            user_triplets = []
            
            # strong > weak
            for pos in strong_books[:5]:
                for neg in weak_books[:3]:
                    user_triplets.append((user_id, pos, neg, 2))
            
            # strong > neg
            for pos in strong_books[:5]:
                for neg in neg_books[:3]:
                    user_triplets.append((user_id, pos, neg, 3))
            
            # weak > neg
            for pos in weak_books[:3]:
                for neg in neg_books[:2]:
                    user_triplets.append((user_id, pos, neg, 1))
            
            if len(user_triplets) > samples_per_user:
                user_triplets = np.random.choice(user_triplets, size=samples_per_user, replace=False).tolist()
            
            triplets.extend(user_triplets)
        
        triplets_df = pd.DataFrame(triplets, columns=['user_id', 'pos_item', 'neg_item', 'weight'])
        print(f"✅ Сгенерировано {len(triplets_df)} триплетов из тренировочных данных")
        
        return triplets_df
    
def clean_column_name(name: str) -> str:
    """Очищает имя колонки для безопасного использования"""
    import re
    # Заменяем точки, пробелы, дефисы на подчеркивания
    cleaned = str(name).replace('.', '_').replace(' ', '_').replace('-', '_')
    # Удаляем другие проблемные символы
    cleaned = ''.join(c if c.isalnum() or c == '_' else '_' for c in cleaned)
    # Убираем множественные подчеркивания
    cleaned = re.sub(r'_+', '_', cleaned)
    # Убираем подчеркивание в начале и конце
    cleaned = cleaned.strip('_')
    return cleaned
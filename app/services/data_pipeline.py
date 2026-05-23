# app/services/data_pipeline.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

class DataLoader:
    """Загрузка и первичная очистка данных"""
    
    def __init__(self, 
                 books_path: str = "data/raw/Books.csv", 
                 ratings_path: str = "data/raw/Ratings.csv", 
                 users_path: str = "data/raw/Users.csv",
                 output_dir: str = "data/processed"):
        
        self.books_path = books_path
        self.ratings_path = ratings_path
        self.users_path = users_path
        self.output_dir = Path(output_dir)
        
        self.books = None
        self.ratings = None
        self.users = None
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Загрузка всех данных"""
        print("📚 Загрузка данных...")
        
        self.books = pd.read_csv(self.books_path, encoding='utf-8', sep=';', on_bad_lines='skip')
        self.ratings = pd.read_csv(self.ratings_path, encoding='utf-8', sep=';', on_bad_lines='skip')
        self.users = pd.read_csv(self.users_path, encoding='utf-8', sep=';', low_memory=False, on_bad_lines='skip')
        
        return {
            'books': self.books,
            'ratings': self.ratings,
            'users': self.users
        }
    
    def clean_data(self, min_book_interactions: int = 15, min_user_interactions: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Очистка данных: приведение типов, фильтрация редких книг и пользователей"""
        
        print("🧹 Очистка данных...")
        
        # Преобразование типов
        self.users['Age'] = pd.to_numeric(self.users['Age'], errors='coerce')
        self.users['User-ID'] = pd.to_numeric(self.users['User-ID'], errors='coerce')
        
        # Фильтрация редких книг (мало взаимодействий)
        book_interaction_counts = self.ratings['ISBN'].value_counts()
        rare_books = book_interaction_counts[book_interaction_counts < min_book_interactions].index.tolist()
        print(f"   Удалено книг с < {min_book_interactions} взаимодействий: {len(rare_books)}")
        
        self.ratings = self.ratings[~self.ratings['ISBN'].isin(rare_books)]
        self.books = self.books[~self.books['ISBN'].isin(rare_books)]
        
        # Фильтрация неактивных пользователей
        user_interaction_counts = self.ratings['User-ID'].value_counts()
        inactive_users = user_interaction_counts[user_interaction_counts < min_user_interactions].index.tolist()
        print(f"   Удалено пользователей с < {min_user_interactions} взаимодействий: {len(inactive_users)}")
        
        self.ratings = self.ratings[~self.ratings['User-ID'].isin(inactive_users)]
        self.users = self.users[~self.users['User-ID'].isin(inactive_users)]
        
        return self.books, self.ratings, self.users
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики по данным"""
        return {
            'n_users': self.ratings['User-ID'].nunique(),
            'n_books': self.ratings['ISBN'].nunique(),
            'n_interactions': len(self.ratings),
            'density': len(self.ratings) / (self.ratings['User-ID'].nunique() * self.ratings['ISBN'].nunique()) * 100
        }
    
    def run_preparation(self) -> Dict[str, Any]:
        """Полный pipeline подготовки данных"""
        self.load_data()
        self.clean_data()
        stats = self.get_statistics()
        
        # Сохраняем очищенные данные в CSV вместо parquet
        self.output_dir.mkdir(parents=True, exist_ok=True)
        clean_ratings_path = self.output_dir / "clean_ratings.csv"
        self.ratings.to_csv(clean_ratings_path, index=False)
        print(f"💾 Очищенные данные сохранены в {clean_ratings_path}")
        
        return {
            "status": "success",
            "n_users": stats['n_users'],
            "n_books": stats['n_books'],
            "n_interactions": stats['n_interactions'],
            "message": f"Данные подготовлены. Плотность матрицы: {stats['density']:.4f}%"
        }


class DataAnalyzer:
    """Анализ данных для принятия решений о разбиении"""
    
    @staticmethod
    def analyze_ratings(ratings_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Анализирует датафрейм с рейтингами и возвращает ключевые статистики
        """
        print("=" * 60)
        print("АНАЛИЗ ДАННЫХ ДЛЯ РЕКОМЕНДАТЕЛЬНОЙ СИСТЕМЫ")
        print("=" * 60)
        
        # Базовые метрики
        metrics = {
            'n_users': ratings_df['User-ID'].nunique(),
            'n_items': ratings_df['ISBN'].nunique(),
            'n_interactions': len(ratings_df),
            'density': len(ratings_df) / (ratings_df['User-ID'].nunique() * ratings_df['ISBN'].nunique()) * 100
        }
        
        # Анализ пользователей
        user_stats = ratings_df.groupby('User-ID').size()
        metrics['user_stats'] = {
            'mean': user_stats.mean(),
            'median': user_stats.median(),
            'min': user_stats.min(),
            'max': user_stats.max()
        }
        
        # Анализ рейтингов
        zero_ratings = ratings_df[ratings_df['Rating'] == 0]
        non_zero_ratings = ratings_df[ratings_df['Rating'] > 0]
        
        metrics['zero_ratings_pct'] = len(zero_ratings) / len(ratings_df) * 100
        
        if len(non_zero_ratings) > 0:
            metrics['avg_rating'] = non_zero_ratings['Rating'].mean()
            metrics['median_rating'] = non_zero_ratings['Rating'].median()
        
        # Вывод статистики
        print(f"\n📊 БАЗОВАЯ СТАТИСТИКА:")
        print(f"   Всего записей: {metrics['n_interactions']:,}")
        print(f"   Пользователей: {metrics['n_users']:,}")
        print(f"   Книг: {metrics['n_items']:,}")
        print(f"   Плотность: {metrics['density']:.4f}%")
        
        print(f"\n👤 АНАЛИЗ ПОЛЬЗОВАТЕЛЕЙ:")
        print(f"   Среднее книг на пользователя: {user_stats.mean():.2f}")
        print(f"   Медиана: {user_stats.median():.2f}")
        
        print(f"\n⭐ АНАЛИЗ РЕЙТИНГОВ:")
        print(f"   Нулевых рейтингов (прочитано без оценки): {metrics['zero_ratings_pct']:.1f}%")
        
        return metrics
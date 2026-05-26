# app/services/collab_filter.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import pickle

from app.services.utils import normalize_author_name, clean_column_name


class CollaborativeFilter:
    """Коллаборативная фильтрация на основе preference_score"""
    
    def __init__(self,
                 strong_pos_threshold: int = 8,
                 weak_pos_is_zero: bool = True,
                 neg_threshold: int = 5,
                 similarity_threshold: float = 0.1):
        
        self.strong_pos_threshold = strong_pos_threshold
        self.weak_pos_is_zero = weak_pos_is_zero
        self.neg_threshold = neg_threshold
        self.similarity_threshold = similarity_threshold
        
        self.user_item_matrix = None
        self.interaction_mask = None
        self.preference_df = None
        self.recommendations = None
    
    def _calculate_preference(self, row: pd.Series) -> float:
        """Расчет preference_score для одного взаимодействия"""
        # Сильно позитивные (явно понравились)
        if row['Rating'] >= self.strong_pos_threshold:
            return 3.0
        # Слабо позитивные (прочитал, но не оценил)
        elif self.weak_pos_is_zero and row['Rating'] == 0:
            return 2.0
        # Негативные (явно не понравились)
        elif row['Rating'] <= self.neg_threshold and row['Rating'] > 0:
            return -3.0
        # Нейтральные (промежуточные рейтинги 6-7)
        else:
            return 1.0
    
    def prepare_preference_data(self, train_ratings: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """
        Подготавливает данные с preference_score для коллаборативной фильтрации
        
        Returns:
            (user_item_matrix, interaction_mask, preference_df)
        """
        print("📊 Подготовка данных preference_score...")
        
        df = train_ratings.copy()
        
        # Расчет preference_score
        df['preference'] = df.apply(self._calculate_preference, axis=1)
        
        # Статистика
        print("Распределение preference_score:")
        print(df['preference'].value_counts().sort_index())
        print(f"\nСредний preference_score: {df['preference'].mean():.3f}")
        
        # Создаем матрицу пользователь-книга
        user_item_matrix = df.pivot(
            index='User-ID', 
            columns='ISBN', 
            values='preference'
        ).fillna(0)
        
        # Маска взаимодействий (для нормализации)
        interaction_mask = (df.pivot(
            index='User-ID', 
            columns='ISBN', 
            values='preference'
        ).notna()).astype(float).fillna(0).values
        
        print(f"\n📐 Размер матрицы: {user_item_matrix.shape}")
        print(f"   Пользователей: {user_item_matrix.shape[0]}")
        print(f"   Книг: {user_item_matrix.shape[1]}")
        print(f"   Плотность: {(user_item_matrix != 0).sum().sum() / user_item_matrix.size * 100:.4f}%")
        
        self.user_item_matrix = user_item_matrix
        self.interaction_mask = interaction_mask
        self.preference_df = df
        
        return user_item_matrix, df, interaction_mask
    
    def generate_candidates(self, 
                           train_ratings: pd.DataFrame,
                           books_df: pd.DataFrame,
                           n_recommendations: int = 1000) -> Dict[int, Dict[str, float]]:
        """
        Генерация кандидатов для всех пользователей
        Теперь включает:
        1. Кандидаты от коллаборативной фильтрации
        2. Кандидаты от популярных авторов (по 2 книги на автора из истории)
        """
        print("\n🚀 Генерация кандидатов коллаборативной фильтрацией...")
        
        # Подготовка данных
        user_item_matrix, _, interaction_mask = self.prepare_preference_data(train_ratings)
        preferences = user_item_matrix.values
        
        # 1. Расчет схожести пользователей
        print("1. Расчет косинусной схожести...")
        user_similarity = cosine_similarity(preferences)
        np.fill_diagonal(user_similarity, 0)
        user_similarity[user_similarity < self.similarity_threshold] = 0
        
        # 2. Взвешенные предсказания
        print("2. Расчет взвешенных предсказаний...")
        weighted_sum = user_similarity.dot(preferences)
        similarity_sum = np.abs(user_similarity).dot(interaction_mask)
        similarity_sum[similarity_sum == 0] = 1
        predictions = weighted_sum / similarity_sum
        predictions[interaction_mask > 0] = -np.inf
        
        # 3. Формирование топ-рекомендаций от CF
        print(f"3. Формирование топ-{n_recommendations} кандидатов от CF...")
        top_n_indices = np.argsort(predictions, axis=1)[:, -n_recommendations:][:, ::-1]
        
        recommendations = {}
        item_ids = user_item_matrix.columns.values
        
        for i, user in enumerate(tqdm(user_item_matrix.index, desc="CF рекомендации")):
            top_items = item_ids[top_n_indices[i]]
            top_scores = predictions[i, top_n_indices[i]]
            valid = top_scores > -np.inf
            if valid.any():
                recommendations[user] = dict(zip(top_items[valid], top_scores[valid]))
        
        # 4. Добавляем кандидатов от авторов из истории пользователя
        print("\n📚 Добавление кандидатов от популярных авторов из истории...")
        recommendations = self._add_author_based_candidates(
            recommendations, train_ratings, books_df
        )
        
        self.recommendations = recommendations
        print(f"\n✅ Сгенерировано кандидатов для {len(recommendations)} пользователей")
        
        return recommendations
    
    def _add_author_based_candidates(self,
                                     recommendations: Dict[int, Dict[str, float]],
                                     train_ratings: pd.DataFrame,
                                     books_df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """
        Добавляет кандидатов от авторов из истории пользователя:
        - Для каждого пользователя находим всех авторов, которых он читал
        - Для каждого автора добавляем 2 самые популярные книги, которые пользователь еще не читал
        """
        # Нормализуем авторов в books_df
        books_df = books_df.copy()
        books_df['Author_norm'] = books_df['Author'].apply(normalize_author_name)
        
        # Считаем популярность книг по авторам (количество прочтений)
        book_popularity = train_ratings.groupby('ISBN').size()
        books_df['book_popularity'] = books_df['ISBN'].map(book_popularity).fillna(0)
        
        # Для каждого автора находим топ-5 самых популярных книг
        author_top_books = {}
        for author in books_df['Author_norm'].unique():
            author_books = books_df[books_df['Author_norm'] == author]
            # Сортируем по популярности и берем топ-5
            top_books = author_books.nlargest(5, 'book_popularity')['ISBN'].tolist()
            author_top_books[author] = top_books
        
        # Нормализуем авторов в train_ratings через books_df
        books_with_authors = books_df[['ISBN', 'Author_norm']].drop_duplicates('ISBN')
        train_with_authors = train_ratings.merge(books_with_authors, on='ISBN', how='left')
        
        # Для каждого пользователя собираем авторов из истории
        user_authors = train_with_authors.groupby('User-ID')['Author_norm'].apply(
            lambda vals: set(vals.dropna())
        ).to_dict()
        
        print(f"   Уникальных авторов в истории: {len(set([a for users in user_authors.values() for a in users]))}")
        
        added_count = 0
        users_with_new_candidates = 0
        
        for user_id, user_recs in tqdm(recommendations.items(), desc="Добавление авторских кандидатов"):
            # Какие книги пользователь уже читал
            user_read_books = set(train_ratings[train_ratings['User-ID'] == user_id]['ISBN'])
            
            # Авторы, которых пользователь читал
            authors_read = user_authors.get(user_id, set())
            
            new_candidates = {}
            
            for author in authors_read:
                # Берем топ-5 книг автора
                top_books = author_top_books.get(author, [])
                # Выбираем книги, которые пользователь еще не читал
                new_books = [book for book in top_books if book not in user_read_books]
                
                # Добавляем по 2 книги от каждого автора
                for book in new_books[:2]:
                    if book not in user_recs:
                        new_candidates[book] = 0
            
            if new_candidates:
                user_recs.update(new_candidates)
                added_count += len(new_candidates)
                users_with_new_candidates += 1
        
        print(f"   Добавлено {added_count} кандидатов от авторов для {users_with_new_candidates} пользователей")

        return recommendations
    
    def save_candidates(self, filepath: str = "models/collaborative_candidates.pkl"):
        """Сохраняет кандидатов на диск"""
        if self.recommendations is None:
            raise ValueError("Нет сгенерированных кандидатов. Сначала вызовите generate_candidates()")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.recommendations, f)
        
        print(f"💾 Кандидаты сохранены в {filepath}")
    
    def load_candidates(self, filepath: str = "models/collaborative_candidates.pkl"):
        """Загружает кандидатов с диска"""
        with open(filepath, 'rb') as f:
            self.recommendations = pickle.load(f)
        
        print(f"📂 Кандидаты загружены из {filepath}")
        return self.recommendations


class PopularityFallback:
    """Фоллбэк с популярными книгами и авторами для малоактивных пользователей"""
    
    @staticmethod
    def get_top_popular_books(train_ratings: pd.DataFrame, n: int = 500) -> List[str]:
        """Возвращает список n самых популярных книг"""
        book_popularity = train_ratings['ISBN'].value_counts()
        top_books = book_popularity.head(n).index.tolist()
        
        print(f"📚 Топ-{n} популярных книг:")
        print(f"   Самая популярная: {top_books[0]} ({book_popularity.iloc[0]} взаимодействий)")
        
        return top_books
    
    @staticmethod
    def get_popular_books_by_author(train_ratings: pd.DataFrame, 
                                     books_df: pd.DataFrame, 
                                     n_books_per_author: int = 3,
                                     top_authors: int = 100) -> Dict[str, List[str]]:
        """
        Возвращает для каждого популярного автора список его самых популярных книг
        """
        # Нормализуем авторов
        books_df = books_df.copy()
        books_df['Author_norm'] = books_df['Author'].apply(normalize_author_name)
        
        # Считаем популярность книг
        book_popularity = train_ratings.groupby('ISBN').size()
        books_df['book_popularity'] = books_df['ISBN'].map(book_popularity).fillna(0)
        
        # Считаем популярность авторов
        author_popularity = books_df.groupby('Author_norm')['book_popularity'].sum().sort_values(ascending=False)
        top_authors_list = author_popularity.head(top_authors).index.tolist()
        
        author_books = {}
        for author in top_authors_list:
            author_books_df = books_df[books_df['Author_norm'] == author]
            top_books = author_books_df.nlargest(n_books_per_author, 'book_popularity')['ISBN'].tolist()
            author_books[author] = top_books
        
        return author_books
    
    @staticmethod
    def get_inactive_users(train_ratings: pd.DataFrame, min_books: int = 20) -> List[int]:
        """Возвращает список пользователей с меньше чем min_books книг"""
        user_activity = train_ratings.groupby('User-ID').size()
        inactive_users = user_activity[user_activity < min_books].index.tolist()
        
        print(f"👤 Пользователей с < {min_books} книгами: {len(inactive_users)} "
              f"({len(inactive_users)/len(user_activity)*100:.1f}%)")
        
        return inactive_users
    
    @staticmethod
    def add_popular_to_inactive(recommendations: Dict, 
                                inactive_users: List[int], 
                                popular_books: List[str], 
                                train_ratings: pd.DataFrame,
                                books_df: pd.DataFrame,
                                add_ratio: float = 0.1,
                                random_seed: int = 42) -> Dict:
        """
        Добавляет случайные популярные книги и книги популярных авторов
        для малоактивных пользователей
        """
        import random
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Получаем популярные книги по авторам
        popular_by_author = PopularityFallback.get_popular_books_by_author(
            train_ratings, books_df, n_books_per_author=3, top_authors=50
        )
        all_popular_by_author_books = []
        for books_list in popular_by_author.values():
            all_popular_by_author_books.extend(books_list)
        
        updated_recommendations = recommendations.copy()
        total_added = 0
        
        for user in tqdm(inactive_users, desc="Добавление популярных книг"):
            if user not in updated_recommendations:
                continue
            
            current_recs = updated_recommendations[user]
            n_current = len(current_recs)
            n_to_add = max(1, int(n_current * add_ratio))
            
            existing_books = set(current_recs.keys())
            
            # Смешиваем обычные популярные книги и книги от популярных авторов
            available_books = [b for b in popular_books if b not in existing_books]
            available_author_books = [b for b in all_popular_by_author_books if b not in existing_books]
            
            # Берем 50% от обычных популярных и 50% от авторских
            n_normal = n_to_add // 2
            n_author = n_to_add - n_normal
            
            chosen_books = []
            if available_books and n_normal > 0:
                chosen_books.extend(random.sample(available_books, min(n_normal, len(available_books))))
            if available_author_books and n_author > 0:
                chosen_books.extend(random.sample(available_author_books, min(n_author, len(available_author_books))))
            
            for book in chosen_books:
                current_recs[book] = 0
            
            total_added += len(chosen_books)
        
        print(f"✅ Добавлено {total_added} популярных книг для {len(inactive_users)} малоактивных пользователей")
        
        return updated_recommendations
    

class CandidateAnalyzer:
    """Анализ качества сгенерированных кандидатов"""
    
    @staticmethod
    def calculate_hit_rate(recommendations: Dict,
                          test_ratings: pd.DataFrame,
                          top_k: Optional[int] = None) -> Dict:
        """
        Рассчитывает hit rate для рекомендаций
        """
        test_users = set(test_ratings['User-ID'].unique())
        hits = []
        
        for user in test_users:
            if user not in recommendations:
                continue
            
            # Тестовые книги пользователя (только strong positive)
            test_books = set(test_ratings[test_ratings['User-ID'] == user]['ISBN'])
            if not test_books:
                continue
            
            # Рекомендации пользователя
            rec_books = recommendations[user]
            if top_k:
                rec_books = dict(list(rec_books.items())[:top_k])
            
            # Hit rate
            hit_count = len(set(rec_books.keys()) & test_books)
            hit_rate = hit_count / len(test_books)
            hits.append(hit_rate)
        
        if not hits:
            return {'error': 'No valid users found'}
        
        results = {
            'mean_hit_rate': np.mean(hits),
            'median_hit_rate': np.median(hits),
            'std_hit_rate': np.std(hits),
            'min_hit_rate': np.min(hits),
            'max_hit_rate': np.max(hits),
            'users_with_hits': sum(1 for h in hits if h > 0) / len(hits),
            'total_users': len(hits),
            'percentile_25': np.percentile(hits, 25),
            'percentile_50': np.percentile(hits, 50),
            'percentile_75': np.percentile(hits, 75),
            'percentile_90': np.percentile(hits, 90)
        }
        
        print(f"\n📊 Hit Rate для {results['total_users']} пользователей (top-{top_k or 'все'}):")
        print(f"   Средний: {results['mean_hit_rate']:.3f}")
        print(f"   Медиана: {results['median_hit_rate']:.3f}")
        print(f"   Пользователей с hits: {results['users_with_hits']:.1%}")
        
        return results
    
    @staticmethod
    def analyze_coverage(recommendations: Dict, test_ratings: pd.DataFrame) -> Dict:
        """
        Анализирует покрытие тестовых пользователей кандидатами
        """
        test_users = set(test_ratings['User-ID'].unique())
        users_with_recs = set(recommendations.keys())
        covered_users = test_users & users_with_recs
        
        return {
            'test_users': len(test_users),
            'users_with_recs': len(users_with_recs),
            'covered_users': len(covered_users),
            'coverage_rate': len(covered_users) / len(test_users) if test_users else 0
        }
# app/services/afm_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import pickle


# app/services/afm_model.py - обновить класс AFMDataset

class AFMDataset(Dataset):
    """Датасет для AFM с фичами пользователей и книг"""
    
    def __init__(self, triplets_df, user_features, book_features, 
                 user_encoder=None, book_encoder=None,
                 user_scaler=None, book_scaler=None,
                 fit_encoders=True):
        
        self.triplets = triplets_df
        
        # Функция для очистки имен от недопустимых символов
        def clean_name(name):
            """Очищает имя колонки от символов, недопустимых в PyTorch module names"""
            # Заменяем точки, пробелы, дефисы на подчеркивания
            cleaned = str(name).replace('.', '_').replace(' ', '_').replace('-', '_')
            # Удаляем другие потенциально проблемные символы
            cleaned = ''.join(c if c.isalnum() or c == '_' else '_' for c in cleaned)
            # Убираем множественные подчеркивания
            import re
            cleaned = re.sub(r'_+', '_', cleaned)
            # Убираем подчеркивание в начале и конце
            cleaned = cleaned.strip('_')
            return cleaned
        
        if fit_encoders:
            # Кодируем ID в индексы
            from sklearn.preprocessing import LabelEncoder
            self.user_encoder = LabelEncoder()
            self.book_encoder = LabelEncoder()
            
            all_users = list(user_features.index) + triplets_df['user_id'].tolist()
            all_books = list(book_features.index) + triplets_df['pos_item'].tolist() + triplets_df['neg_item'].tolist()
            
            self.user_encoder.fit(all_users)
            self.book_encoder.fit(all_books)
        else:
            self.user_encoder = user_encoder
            self.book_encoder = book_encoder
        
        # Подготовка фичей
        self.user_features = user_features
        self.book_features = book_features
        
        # Определяем числовые колонки
        self.user_numerical_cols = ['user_rating_count', 'non_zero_ratings_count', 'zero_ratings_count', 
                                    'non_zero_ratio', 'user_avg_rating', 'user_rating_std',
                                    'user_rating_range', 'user_rating_variability']
        self.book_numerical_cols = ['book_rating_count', 'book_avg_rating', 'book_rating_std', 
                                    'popularity_norm', 'wilson_score', 'book_age', 'publisher_book_count']
        
        # Фильтруем только существующие колонки
        self.user_numerical_cols = [col for col in self.user_numerical_cols if col in user_features.columns]
        self.book_numerical_cols = [col for col in self.book_numerical_cols if col in book_features.columns]
        
        # Нормализация
        from sklearn.preprocessing import StandardScaler
        
        if fit_encoders:
            self.user_scaler = StandardScaler()
            self.book_scaler = StandardScaler()
            
            # Обучаем scaler на всех данных
            self.user_scaler.fit(user_features[self.user_numerical_cols].fillna(0).values)
            self.book_scaler.fit(book_features[self.book_numerical_cols].fillna(0).values)
        else:
            self.user_scaler = user_scaler
            self.book_scaler = book_scaler
        
        # Категориальные фичи с очищенными именами
        # Для пользователей
        raw_user_cat_cols = [col for col in user_features.columns 
                            if col.startswith('activity_') or col.startswith('age_')]
        
        # Для книг
        raw_book_cat_cols = [col for col in book_features.columns 
                            if col.startswith('Author_popular_') or 
                               col.startswith('Publisher_popular_') or 
                               col.startswith('Year_era_') or
                               col in ['is_classic', 'is_popular_author']]
        
        # Очищаем имена и сохраняем соответствие
        self.user_cat_cols = [clean_name(col) for col in raw_user_cat_cols]
        self.book_cat_cols = [clean_name(col) for col in raw_book_cat_cols]
        
        # Словари для соответствия очищенных имен оригинальным
        self.user_col_mapping = {clean_name(col): col for col in raw_user_cat_cols}
        self.book_col_mapping = {clean_name(col): col for col in raw_book_cat_cols}
        
        # Словари для размеров эмбеддингов
        self.user_cat_dims = {col: 2 for col in self.user_cat_cols}
        self.book_cat_dims = {col: 2 for col in self.book_cat_cols}
        
        # Сохраняем индексы колонок для быстрого доступа
        self.user_cat_indices = {col: i for i, col in enumerate(self.user_cat_cols)}
        self.book_cat_indices = {col: i for i, col in enumerate(self.book_cat_cols)}
        
        print(f"   User categories ({len(self.user_cat_cols)}): {self.user_cat_cols[:5]}...")
        print(f"   Book categories ({len(self.book_cat_cols)}): {self.book_cat_cols[:5]}...")
    
    def __len__(self):
        return len(self.triplets)
    

    def __getitem__(self, idx):
        row = self.triplets.iloc[idx]
        
        # Кодируем ID
        user_id = self.user_encoder.transform([row['user_id']])[0]
        pos_id = self.book_encoder.transform([row['pos_item']])[0]
        neg_id = self.book_encoder.transform([row['neg_item']])[0]
        
        # Получаем фичи пользователя
        if row['user_id'] in self.user_features.index:
            user_f = self.user_features.loc[row['user_id']]
        else:
            user_f = pd.Series(index=self.user_features.columns, dtype=float).fillna(0)
        
        user_numerical = torch.tensor(
            self.user_scaler.transform([user_f[self.user_numerical_cols].fillna(0).values])[0], 
            dtype=torch.float32
        )
        
        # Категориальные фичи пользователя
        user_categorical = []
        for clean_col in self.user_cat_cols:
            orig_col = self.user_col_mapping.get(clean_col, clean_col)
            # Исправленная проверка
            if orig_col in user_f.index:
                val = user_f[orig_col]
                # Проверяем, является ли val скаляром или Series
                if hasattr(val, 'iloc'):
                    val = val.iloc[0] if len(val) > 0 else 0
                # Проверяем на NaN
                if pd.isna(val):
                    val = 0
            else:
                val = 0
            user_categorical.append(int(val))
        user_categorical = torch.tensor(user_categorical, dtype=torch.long)
        
        # Получаем фичи для pos книги
        if row['pos_item'] in self.book_features.index:
            pos_f = self.book_features.loc[row['pos_item']]
        else:
            pos_f = pd.Series(index=self.book_features.columns, dtype=float).fillna(0)
        
        pos_numerical = torch.tensor(
            self.book_scaler.transform([pos_f[self.book_numerical_cols].fillna(0).values])[0], 
            dtype=torch.float32
        )
        
        pos_categorical = []
        for clean_col in self.book_cat_cols:
            orig_col = self.book_col_mapping.get(clean_col, clean_col)
            # Исправленная проверка
            if orig_col in pos_f.index:
                val = pos_f[orig_col]
                if hasattr(val, 'iloc'):
                    val = val.iloc[0] if len(val) > 0 else 0
                if pd.isna(val):
                    val = 0
            else:
                val = 0
            pos_categorical.append(int(val))
        pos_categorical = torch.tensor(pos_categorical, dtype=torch.long)
        
        # Получаем фичи для neg книги
        if row['neg_item'] in self.book_features.index:
            neg_f = self.book_features.loc[row['neg_item']]
        else:
            neg_f = pd.Series(index=self.book_features.columns, dtype=float).fillna(0)
        
        neg_numerical = torch.tensor(
            self.book_scaler.transform([neg_f[self.book_numerical_cols].fillna(0).values])[0], 
            dtype=torch.float32
        )
        
        neg_categorical = []
        for clean_col in self.book_cat_cols:
            orig_col = self.book_col_mapping.get(clean_col, clean_col)
            # Исправленная проверка
            if orig_col in neg_f.index:
                val = neg_f[orig_col]
                if hasattr(val, 'iloc'):
                    val = val.iloc[0] if len(val) > 0 else 0
                if pd.isna(val):
                    val = 0
            else:
                val = 0
            neg_categorical.append(int(val))
        neg_categorical = torch.tensor(neg_categorical, dtype=torch.long)
        
        weight = torch.tensor(float(row['weight']), dtype=torch.float32)
        
        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'pos_id': torch.tensor(pos_id, dtype=torch.long),
            'neg_id': torch.tensor(neg_id, dtype=torch.long),
            'user_numerical': user_numerical,
            'user_categorical': user_categorical,
            'pos_numerical': pos_numerical,
            'neg_numerical': neg_numerical,
            'pos_categorical': pos_categorical,
            'neg_categorical': neg_categorical,
            'weight': weight
        }


class AFMRanker(nn.Module):
    """Attentional Factorization Machine для ранжирования"""
    
    def __init__(self, user_count, book_count, user_cat_dims, book_cat_dims,
                 user_num_dim, book_num_dim, embed_dim=64, attention_dim=32, dropout=0.2):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Эмбеддинги для ID
        self.user_embed = nn.Embedding(user_count, embed_dim)
        self.book_embed = nn.Embedding(book_count, embed_dim)
        
        # Эмбеддинги для категориальных фичей пользователя
        self.user_cat_embeds = nn.ModuleDict({
            name: nn.Embedding(dim, embed_dim) 
            for name, dim in user_cat_dims.items()
        })
        
        # Эмбеддинги для категориальных фичей книги
        self.book_cat_embeds = nn.ModuleDict({
            name: nn.Embedding(dim, embed_dim) 
            for name, dim in book_cat_dims.items()
        })
        
        # Линейные слои для числовых фичей
        self.user_num_linear = nn.Linear(user_num_dim, embed_dim)
        self.book_num_linear = nn.Linear(book_num_dim, embed_dim)
        
        # Attention network
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 1)
        )
        
        # Финальный слой для предсказания
        self.predict = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )
        
    def get_embedding(self, user_id, book_id, user_num, user_cat, book_num, book_cat):
        """Получает объединенный эмбеддинг для пары (user, book)"""
        
        # Эмбеддинги ID
        user_emb = self.user_embed(user_id)
        book_emb = self.book_embed(book_id)
        
        # Эмбеддинги категориальных фичей пользователя
        if len(self.user_cat_embeds) > 0:
            user_cat_emb_list = []
            for i, name in enumerate(self.user_cat_embeds.keys()):
                emb = self.user_cat_embeds[name](user_cat[:, i])
                user_cat_emb_list.append(emb)
            user_cat_emb = torch.stack(user_cat_emb_list, dim=1).mean(dim=1)
        else:
            user_cat_emb = torch.zeros_like(user_emb)
        
        # Эмбеддинги категориальных фичей книги
        if len(self.book_cat_embeds) > 0:
            book_cat_emb_list = []
            for i, name in enumerate(self.book_cat_embeds.keys()):
                emb = self.book_cat_embeds[name](book_cat[:, i])
                book_cat_emb_list.append(emb)
            book_cat_emb = torch.stack(book_cat_emb_list, dim=1).mean(dim=1)
        else:
            book_cat_emb = torch.zeros_like(book_emb)
        
        # Числовые фичи
        user_num_emb = self.user_num_linear(user_num)
        book_num_emb = self.book_num_linear(book_num)
        
        # Объединяем
        combined = user_emb + book_emb + user_cat_emb + book_cat_emb + user_num_emb + book_num_emb
        return combined
    
    def forward(self, batch):
        # Эмбеддинги для позитивной пары
        pos_emb = self.get_embedding(
            batch['user_id'], batch['pos_id'],
            batch['user_numerical'], batch['user_categorical'],
            batch['pos_numerical'], batch['pos_categorical']
        )
        
        # Эмбеддинги для негативной пары
        neg_emb = self.get_embedding(
            batch['user_id'], batch['neg_id'],
            batch['user_numerical'], batch['user_categorical'],
            batch['neg_numerical'], batch['neg_categorical']
        )
        
        # Attention веса
        pos_attn = torch.sigmoid(self.attention(pos_emb))
        neg_attn = torch.sigmoid(self.attention(neg_emb))
        
        # Применяем веса
        pos_weighted = pos_emb * pos_attn
        neg_weighted = neg_emb * neg_attn
        
        # Предсказания
        pos_score = self.predict(pos_weighted).squeeze()
        neg_score = self.predict(neg_weighted).squeeze()
        
        # BPR loss
        loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8)
        weighted_loss = (loss * batch['weight']).mean()
        
        return weighted_loss, pos_score, neg_score
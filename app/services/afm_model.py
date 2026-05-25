import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

class AFMRanker(nn.Module):
    """Правильная реализация Attentive Factorization Machine"""
    
    def __init__(self, user_count, book_count, 
                 user_num_dim, book_num_dim,
                 embed_dim=64, attention_dim=32, dropout=0.2):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Эмбеддинги для ID
        self.user_embed = nn.Embedding(user_count, embed_dim)
        self.book_embed = nn.Embedding(book_count, embed_dim)
        
        # Linear terms
        self.user_linear = nn.Embedding(user_count, 1)
        self.book_linear = nn.Embedding(book_count, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Проекции для числовых фичей
        self.user_num_proj = nn.Linear(user_num_dim, embed_dim) if user_num_dim > 0 else None
        self.book_num_proj = nn.Linear(book_num_dim, embed_dim) if book_num_dim > 0 else None
        
        # Линейные веса для числовых фичей
        self.user_num_linear = nn.Linear(user_num_dim, 1) if user_num_dim > 0 else None
        self.book_num_linear = nn.Linear(book_num_dim, 1) if book_num_dim > 0 else None
        
        # Attention network
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 1, bias=False)
        )
        
        # Финальный слой
        self.dropout = nn.Dropout(dropout)
        self.final_layer = nn.Linear(embed_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
    
    def forward(self, batch):
        """Forward для обучения (принимает batch словарь)"""
        user_ids = batch['user_id']
        pos_ids = batch['pos_id']
        neg_ids = batch['neg_id']
        user_num = batch.get('user_numerical')
        pos_num = batch.get('pos_numerical')
        neg_num = batch.get('neg_numerical')
        
        pos_score = self._predict_score(user_ids, pos_ids, user_num, pos_num)
        neg_score = self._predict_score(user_ids, neg_ids, user_num, neg_num)
        
        loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8)
        
        if 'weight' in batch:
            loss = loss * batch['weight']
        
        return loss.mean(), pos_score, neg_score
    
    def _predict_score(self, user_ids, book_ids, user_num=None, book_num=None):
        """Внутренний метод для предсказания score"""
        batch_size = user_ids.size(0)
        
        # Linear part
        linear_score = (self.user_linear(user_ids) + 
                       self.book_linear(book_ids) + 
                       self.global_bias).squeeze()
        
        if user_num is not None and self.user_num_linear is not None:
            linear_score += self.user_num_linear(user_num).squeeze()
        if book_num is not None and self.book_num_linear is not None:
            linear_score += self.book_num_linear(book_num).squeeze()
        
        # Собираем эмбеддинги
        embeddings = [self.user_embed(user_ids), self.book_embed(book_ids)]
        
        if user_num is not None and self.user_num_proj is not None:
            embeddings.append(self.user_num_proj(user_num))
        if book_num is not None and self.book_num_proj is not None:
            embeddings.append(self.book_num_proj(book_num))
        
        if len(embeddings) <= 2:
            return linear_score
        
        all_embeddings = torch.stack(embeddings, dim=1)
        n_features = all_embeddings.size(1)
        
        # Попарные взаимодействия
        pairwise_interactions = []
        attention_weights = []
        
        for i in range(n_features):
            for j in range(i+1, n_features):
                interaction = all_embeddings[:, i] * all_embeddings[:, j]
                pairwise_interactions.append(interaction)
                attention_weights.append(self.attention(interaction))
        
        pairwise_stack = torch.stack(pairwise_interactions, dim=1)
        attention_stack = torch.stack(attention_weights, dim=1)
        
        attention_weights_norm = F.softmax(attention_stack, dim=1)
        weighted_sum = (attention_weights_norm * pairwise_stack).sum(dim=1)
        
        final_score = linear_score + self.final_layer(self.dropout(weighted_sum)).squeeze()
        
        return final_score
    
    def forward_single(self, user_id, book_id, user_num=None, book_num=None):
        """Forward для одного предсказания (инференс)"""
        if not isinstance(user_id, torch.Tensor):
            device = next(self.parameters()).device
            user_id = torch.tensor([user_id], device=device)
            book_id = torch.tensor([book_id], device=device)
        
        return self._predict_score(user_id, book_id, user_num, book_num)
    

class AFMDataset(Dataset):
    """Dataset для AFM с поддержкой fit_encoders параметра"""
    
    def __init__(self, triplets_df, user_features, book_features, 
                 user_encoder=None, book_encoder=None,
                 user_scaler=None, book_scaler=None,
                 fit_encoders=True):
        
        self.triplets = triplets_df
        self.user_features = user_features
        self.book_features = book_features
        
        # Опциональные параметры (для совместимости)
        self.user_cat_cols = []
        self.book_cat_cols = []
        self.user_cat_dims = {}
        self.book_cat_dims = {}
        
        if fit_encoders:
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            
            # Кодируем ID
            self.user_encoder = LabelEncoder()
            self.book_encoder = LabelEncoder()
            
            all_users = list(user_features.index) + triplets_df['user_id'].tolist()
            all_books = list(book_features.index) + triplets_df['pos_item'].tolist() + triplets_df['neg_item'].tolist()

            all_user_cols = user_features.columns.tolist()
            all_book_cols = book_features.columns.tolist()

            print(f"\n👤 USER FEATURES:")
            print(f"   Всего колонок: {len(all_user_cols)}")
            print(f"   Типы колонок:")
            print(f"      Числовые: {len(user_features.select_dtypes(include=[np.number]).columns)}")
            print(f"      Категориальные: {len(user_features.select_dtypes(include=['object', 'category']).columns)}")
            print(f"   Первые 10 колонок: {all_user_cols[:10]}")
            
            print(f"\n📚 BOOK FEATURES:")
            print(f"   Всего колонок: {len(all_book_cols)}")
            print(f"   Типы колонок:")
            print(f"      Числовые: {len(book_features.select_dtypes(include=[np.number]).columns)}")
            print(f"      Категориальные: {len(book_features.select_dtypes(include=['object', 'category']).columns)}")
            print(f"   Первые 10 колонок: {all_book_cols[:10]}")
            
            self.user_encoder.fit(all_users)
            self.book_encoder.fit(all_books)
            
            # Определяем числовые колонки
            self.user_num_cols = user_features.select_dtypes(include=[np.number]).columns.tolist()
            self.book_num_cols = book_features.select_dtypes(include=[np.number]).columns.tolist()
            
            # Фильтруем только существующие колонки
            self.user_num_cols = [col for col in self.user_num_cols if col not in ['User-ID', 'user_id', 'ISBN']]
            self.book_num_cols = [col for col in self.book_num_cols if col not in ['ISBN', 'index']]

            print(f"\n✅ ПРИЗНАКИ ДЛЯ ОБУЧЕНИЯ:")
            print(f"   User числовых: {len(self.user_num_cols)}")
            print(f"   Book числовых: {len(self.book_num_cols)}")
            print(f"   Всего признаков на входе модели: {len(self.user_num_cols) + len(self.book_num_cols)}")
            
            if len(self.user_num_cols) > 0:
                print(f"   Примеры user признаков: {self.user_num_cols[:10]}")
            if len(self.book_num_cols) > 0:
                print(f"   Примеры book признаков: {self.book_num_cols[:10]}")

            # Проверяем, не потеряли ли мы важные признаки
            important_user_cols = ['user_rating_count', 'user_avg_rating', 'user_rating_std', 
                                'non_zero_ratio', 'user_activity_level']
            important_book_cols = ['book_rating_count', 'book_avg_rating', 'popularity_norm',
                                'book_age', 'publisher_book_count']
            
            missing_user = [col for col in important_user_cols if col in all_user_cols and col not in self.user_num_cols]
            missing_book = [col for col in important_book_cols if col in all_book_cols and col not in self.book_num_cols]
            
            if missing_user:
                print(f"\n⚠️  ВНИМАНИЕ: Потеряны важные user признаки: {missing_user}")
            if missing_book:
                print(f"⚠️  ВНИМАНИЕ: Потеряны важные book признаки: {missing_book}")

            # Нормализация числовых фичей
            self.user_scaler = StandardScaler()
            self.book_scaler = StandardScaler()
            
            # Обучаем scaler'ы
            if self.user_num_cols:
                self.user_scaler.fit(user_features[self.user_num_cols].fillna(0).values)
            if self.book_num_cols:
                self.book_scaler.fit(book_features[self.book_num_cols].fillna(0).values)
        else:
            self.user_encoder = user_encoder
            self.book_encoder = book_encoder
            self.user_scaler = user_scaler
            self.book_scaler = book_scaler
            self.user_num_cols = user_features.select_dtypes(include=[np.number]).columns.tolist()
            self.book_num_cols = book_features.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"   User numerical features ({len(self.user_num_cols)}): {self.user_num_cols}...")
        print(f"   Book numerical features ({len(self.book_num_cols)}): {self.book_num_cols}...")
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        row = self.triplets.iloc[idx]
        
        # Кодируем ID
        user_id = self.user_encoder.transform([row['user_id']])[0]
        pos_id = self.book_encoder.transform([row['pos_item']])[0]
        neg_id = self.book_encoder.transform([row['neg_item']])[0]
        
        # Получаем ВСЕ числовые фичи
        if row['user_id'] in self.user_features.index:
            user_row = self.user_features.loc[row['user_id']]
            user_num_arr = self.user_scaler.transform([user_row[self.user_num_cols].fillna(0).values])[0]
        else:
            user_num_arr = np.zeros(len(self.user_num_cols))
        
        if row['pos_item'] in self.book_features.index:
            pos_row = self.book_features.loc[row['pos_item']]
            pos_num_arr = self.book_scaler.transform([pos_row[self.book_num_cols].fillna(0).values])[0]
        else:
            pos_num_arr = np.zeros(len(self.book_num_cols))
        
        if row['neg_item'] in self.book_features.index:
            neg_row = self.book_features.loc[row['neg_item']]
            neg_num_arr = self.book_scaler.transform([neg_row[self.book_num_cols].fillna(0).values])[0]
        else:
            neg_num_arr = np.zeros(len(self.book_num_cols))
        
        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'pos_id': torch.tensor(pos_id, dtype=torch.long),
            'neg_id': torch.tensor(neg_id, dtype=torch.long),
            'user_numerical': torch.tensor(user_num_arr, dtype=torch.float32),
            'pos_numerical': torch.tensor(pos_num_arr, dtype=torch.float32),
            'neg_numerical': torch.tensor(neg_num_arr, dtype=torch.float32),
            'weight': torch.tensor(row.get('weight', 1.0), dtype=torch.float32)
        }
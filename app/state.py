# app/state.py
"""
Модуль для хранения глобального состояния приложения.
Просто контейнер для переменных, без лишней логики.
"""

from typing import Optional
import pandas as pd
from app.services.train_model import DataCache

# Инициализируем кэш
data_cache = DataCache()

# Глобальные переменные (заполняются при старте)
model = None
candidates = None
train_ratings = None
test_ratings = None
book_features = None
user_features = None
triplets = None
books_df = None
books_titles_dict = None
shap_analyzer = None

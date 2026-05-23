import re
import pandas as pd

def normalize_author_name(name):
    """
    Нормализует имя автора:
    - Приводит к нижнему регистру
    - Убирает лишние пробелы
    - Убирает точки и запятые
    - Убирает суффиксы типа 'Jr.', 'Sr.' и т.д.
    """
    if pd.isna(name):
        return 'unknown'
    
    # Приводим к нижнему регистру
    name = str(name).lower().strip()
    
    # Убираем точки и запятые
    name = re.sub(r'[.,]', '', name)
    
    # Убираем суффиксы
    name = re.sub(r'\s+jr\.?$', '', name)
    name = re.sub(r'\s+sr\.?$', '', name)
    name = re.sub(r'\s+ii+$', '', name)
    name = re.sub(r'\s+iii+$', '', name)
    
    # Убираем лишние пробелы
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

def clean_column_name(name: str) -> str:
    """Очищает имя колонки для безопасного использования"""
    # Заменяем точки, пробелы, дефисы на подчеркивания
    cleaned = str(name).replace('.', '_').replace(' ', '_').replace('-', '_')
    # Удаляем другие проблемные символы
    cleaned = ''.join(c if c.isalnum() or c == '_' else '_' for c in cleaned)
    # Убираем множественные подчеркивания
    cleaned = re.sub(r'_+', '_', cleaned)
    # Убираем подчеркивание в начале и конце
    cleaned = cleaned.strip('_')
    return cleaned

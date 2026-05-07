from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Пути
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    raw_data_path: Path = Path("data/raw/books.parquet")
    processed_data_path: Path = Path("data/processed/clean_books.parquet")
    model_path: Path = Path("models/afm_model.pth")
    
    # Параметры модели
    test_size: float = 0.2
    random_state: int = 42
    n_candidates: int = 100

    # Настройки модели
    embed_dim: int = 64
    attention_dim: int = 32
    dropout: float = 0.2
    batch_size: int = 512
    training_epochs: int = 10
    learning_rate: float = 0.001

    class Config:
        env_file = ".env"

settings = Settings()
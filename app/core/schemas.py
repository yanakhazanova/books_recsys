from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class PrepareDataResponse(BaseModel):
    status: str
    n_users: int
    n_books: int
    n_interactions: int
    message: str

class TrainResponse(BaseModel):
    status: str
    model_path: str
    cv_score: Optional[float] = None

class GlobalShapResponse(BaseModel):
    feature_importance: Dict[str, float]  # feature_name -> mean(|SHAP|)
    shap_values_shape: List[int]
    message: str

class MetricsResponse(BaseModel):
    ndcg_at_k: float
    precision_at_k: float
    recall_at_k: float
    k: int
    hit_rate_at_k: Optional[float] = None
    users_with_hits_ratio: Optional[float] = None
    sample_size: Optional[int] = None
    total_users: Optional[int] = None

class UserRecommendation(BaseModel):
    book_id: int
    book_title: Optional[str] = None
    score: float
    rank: Optional[int] = None

class UserRecsResponse(BaseModel):
    user_id: int
    recommendations: List[UserRecommendation]
    shap_interpretation: Dict[str, float]  # feature_name -> SHAP value for this user
    explanation_text: str

class SplitParams(BaseModel):
    strong_pos_threshold: int = 8
    weak_pos_is_zero: bool = True
    neg_threshold: int = 5
    test_items_per_user: int = 2
    min_strong_pos: int = 3
    random_state: int = 42

class CandidatesResponse(BaseModel):
    status: str
    n_users_with_candidates: int
    avg_candidates_per_user: float
    candidates_path: str
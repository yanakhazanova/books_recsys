# Заглушка
def global_importance():
    return {
        "feature_importance": {
            "collab_score": 0.45,
            "book_popularity": 0.32,
            "user_avg_rating": 0.18,
            "genre_match": 0.05
        },
        "shap_values_shape": [100, 4],
        "message": "Глобальная SHAP интерпретация (заглушка)"
    }
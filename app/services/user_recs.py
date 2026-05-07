# Заглушка
def get_user_recommendations(user_id, n_recommendations=10):
    recs = [
        {"book_id": i, "book_title": f"Book {i}", "score": 0.95 - i*0.05}
        for i in range(1, n_recommendations+1)
    ]
    shap_dict = {
        "collab_score": 0.32,
        "book_popularity": 0.15,
        "user_avg_rating": -0.08,
        "genre_match": 0.12
    }
    return recs, shap_dict

def format_shap_explanation(shap_dict):
    pos_features = [f for f, v in shap_dict.items() if v > 0]
    neg_features = [f for f, v in shap_dict.items() if v < 0]
    return f"Положительно повлияли: {', '.join(pos_features)}. Отрицательно: {', '.join(neg_features)}."
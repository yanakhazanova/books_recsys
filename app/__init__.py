# Это временные заглушки, чтобы приложение запускалось
# Позже заменим на реальную реализацию

class data_pipeline:
    @staticmethod
    def run_preparation():
        return {
            "status": "success",
            "n_users": 1000,
            "n_books": 5000,
            "n_interactions": 50000,
            "message": "Данные подготовлены (заглушка)"
        }
    
    @staticmethod
    def split_train_test():
        return "train_data", "test_data"

class collab_filter:
    @staticmethod
    def generate_candidates(train, test):
        return "candidates_df"

class feature_eng:
    @staticmethod
    def build_features(candidates):
        return "features_df"

class pairwise_data:
    @staticmethod
    def create_pairwise_dataset(features):
        return "pairwise_dataset"

class train_model:
    @staticmethod
    def train(pairwise_dataset):
        return "models/ranker.pkl"

class metrics:
    @staticmethod
    def calculate_metrics(k=10):
        return {
            "ndcg_at_k": 0.42,
            "precision_at_k": 0.35,
            "recall_at_k": 0.28
        }

class shap_analysis:
    @staticmethod
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

class user_recs:
    @staticmethod
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
    
    @staticmethod
    def format_shap_explanation(shap_dict):
        pos_features = [f for f, v in shap_dict.items() if v > 0]
        neg_features = [f for f, v in shap_dict.items() if v < 0]
        return f"Положительно повлияли: {', '.join(pos_features)}. Отрицательно: {', '.join(neg_features)}."
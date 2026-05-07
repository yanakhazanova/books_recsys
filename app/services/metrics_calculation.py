import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import Dict, List, Tuple


class MetricsCalculator:
    """Расчет метрик качества рекомендаций"""
    
    @staticmethod
    def calculate_hit_rate(recommendations: Dict[int, List[str]], 
                          test_ratings: pd.DataFrame, 
                          k: int = 10) -> Dict:
        """
        Расчет Hit Rate@k
        
        Hit Rate - доля пользователей, у которых хотя бы одна тестовая книга попала в рекомендации
        """
        test_users = set(test_ratings['User-ID'].unique())
        
        hit_counts = []
        users_with_hits = 0
        
        for user in test_users:
            if user not in recommendations:
                continue
            
            test_books = set(test_ratings[test_ratings['User-ID'] == user]['ISBN'])
            if not test_books:
                continue
            
            rec_books = set(recommendations[user][:k])
            hits = len(rec_books & test_books)
            
            if hits > 0:
                users_with_hits += 1
            
            hit_counts.append(hits / len(test_books))
        
        return {
            'mean_hit_rate': np.mean(hit_counts) if hit_counts else 0,
            'users_with_hits_ratio': users_with_hits / len(test_users) if test_users else 0,
            'total_users_evaluated': len(hit_counts),
            'test_users_total': len(test_users)
        }
    
    @staticmethod
    def calculate_ndcg(recommendations: Dict[int, List[str]], 
                      test_ratings: pd.DataFrame, 
                      k: int = 10) -> float:
        """
        Расчет NDCG@k (Normalized Discounted Cumulative Gain)
        """
        test_users = set(test_ratings['User-ID'].unique())
        ndcg_scores = []
        
        for user in test_users:
            if user not in recommendations:
                continue
            
            test_books = set(test_ratings[test_ratings['User-ID'] == user]['ISBN'])
            if not test_books:
                continue
            
            rec_books = recommendations[user][:k]
            
            # DCG
            dcg = 0
            for i, book in enumerate(rec_books):
                if book in test_books:
                    dcg += 1 / np.log2(i + 2)
            
            # IDCG (идеальный случай)
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(test_books), k)))
            
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_scores.append(ndcg)
        
        return np.mean(ndcg_scores) if ndcg_scores else 0
    
    @staticmethod
    def calculate_precision_recall(recommendations: Dict[int, List[str]], 
                                   test_ratings: pd.DataFrame, 
                                   k: int = 10) -> Dict:
        """
        Расчет Precision@k и Recall@k
        """
        test_users = set(test_ratings['User-ID'].unique())
        
        precisions = []
        recalls = []
        
        for user in test_users:
            if user not in recommendations:
                continue
            
            test_books = set(test_ratings[test_ratings['User-ID'] == user]['ISBN'])
            if not test_books:
                continue
            
            rec_books = set(recommendations[user][:k])
            hits = len(rec_books & test_books)
            
            # Precision@k
            precision = hits / k
            precisions.append(precision)
            
            # Recall@k
            recall = hits / len(test_books)
            recalls.append(recall)
        
        return {
            'precision_at_k': np.mean(precisions) if precisions else 0,
            'recall_at_k': np.mean(recalls) if recalls else 0
        }
    
    @staticmethod
    def calculate_all_metrics(recommendations: Dict[int, List[str]], 
                             test_ratings: pd.DataFrame, 
                             k: int = 10) -> Dict:
        """
        Рассчитывает все метрики
        """
        print(f"\n📊 Расчет метрик @{k}...")
        
        hit_rate_metrics = MetricsCalculator.calculate_hit_rate(recommendations, test_ratings, k)
        ndcg = MetricsCalculator.calculate_ndcg(recommendations, test_ratings, k)
        pr_metrics = MetricsCalculator.calculate_precision_recall(recommendations, test_ratings, k)
        
        return {
            'k': k,
            'hit_rate': hit_rate_metrics['mean_hit_rate'],
            'users_with_hits_ratio': hit_rate_metrics['users_with_hits_ratio'],
            'ndcg_at_k': ndcg,
            'precision_at_k': pr_metrics['precision_at_k'],
            'recall_at_k': pr_metrics['recall_at_k'],
            'total_users_evaluated': hit_rate_metrics['total_users_evaluated'],
            'test_users_total': hit_rate_metrics['test_users_total']
        }

from finsentiment.datasets.sentiment import FinancialSentimentModel
from finsentiment.datasets.preprocessing import prepare_combined_dataset

def get_dataset_class(multi_task=False):
    """Factory function to return the correct dataset class."""
    return FinancialSentimentModel

__all__ = [
    'FinancialSentimentModel', 
    'prepare_combined_dataset',
    'get_dataset_class'
]
from finsentiment.datasets.dataset_single import FinancialSentimentDataset
from finsentiment.datasets.dataset_multi import MultiTaskDataset
from finsentiment.datasets.preprocessing import prepare_combined_dataset

def get_dataset_class(multi_task=False):
    """Factory function to return the correct dataset class."""
    if multi_task:
        return MultiTaskDataset
    return FinancialSentimentDataset

__all__ = [
    'FinancialSentimentDataset', 
    'MultiTaskDataset', 
    'prepare_combined_dataset',
    'get_dataset_class'
]
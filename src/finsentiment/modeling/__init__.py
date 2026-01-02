from finsentiment.modeling.single_task import FinancialSentimentModel
from finsentiment.modeling.multi_task import MultiTaskSentimentModel

def get_model_class(multi_task=False):
    if multi_task:
        return MultiTaskSentimentModel
    return FinancialSentimentModel

__all__ = ['FinancialSentimentModel', 'MultiTaskSentimentModel', 'get_model_class']

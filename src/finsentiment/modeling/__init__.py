from finsentiment.modeling.bert import FinancialSentimentModel

def get_model_class(multi_task=False):

    return FinancialSentimentModel

__all__ = ['FinancialSentimentModel', 'get_model_class']

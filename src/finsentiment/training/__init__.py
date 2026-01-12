from finsentiment.training.trainer import train_multi_task_model

def get_trainer_function(multi_task=False):
    if multi_task:
        return train_multi_task_model
    return train_multi_task_model

__all__ = ['train_model', 'train_multi_task_model', 'get_trainer_function']

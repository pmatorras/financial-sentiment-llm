"""Multi-task dataset that handles both classification and regression."""

import torch
import pandas as pd
from torch.utils.data import Dataset

class FinancialSentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Get text
        text = row.get('text', row.get('sentence', ''))
        if pd.isna(text):
            text = ""
        text = str(text)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Task type and labels
        task_type = row.get('task_type', 'classification')
        
        if task_type == 'regression':
            # FiQA: use continuous score (already in [-1, 1])
            target = torch.tensor(row['score'], dtype=torch.float)
        else:
            # Classification: discrete label
            target = torch.tensor(row['label'], dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': target,
            'task_type': task_type
        }

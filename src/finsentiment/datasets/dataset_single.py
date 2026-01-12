"""PyTorch Dataset classes."""

import torch
import pandas as pd
from torch.utils.data import Dataset

class FinancialSentimentDataset(Dataset):
    """PyTorch dataset for financial sentiment."""
    
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Debug: Check what columns we have
        print(f"Dataset columns: {self.data.columns.tolist()}")
        print(f"First row sample: {self.data.iloc[0].to_dict()}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Try to get text - adjust column name if needed
        if 'text' in row:
            text = row['text']
        # Convert to string and handle None/NaN
        if pd.isna(text) or text is None:
            text = ""
        else:
            text = str(text)
        
        label = int(row['label'])
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

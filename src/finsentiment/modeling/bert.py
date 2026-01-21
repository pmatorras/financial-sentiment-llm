"""Multi-task model with dual heads for classification + regression."""

import torch.nn as nn
from transformers import AutoModel
from finsentiment.config import MODEL_NAME

class FinancialSentimentModel(nn.Module):
    def __init__(self, model_name=None, num_classes=3):
        super().__init__()
        if model_name is None:
            model_name = MODEL_NAME
        
        # Shared BERT encoder
        self.encoder = AutoModel.from_pretrained(
            model_name,
            use_safetensors=True
        )
        
        # Get hidden size from encoder config
        hidden_size = self.encoder.config.hidden_size
        
        # Classification head (for PhraseBank, Twitter)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes)
        )
        
        # Regression head (for FiQA continuous scores)
        self.regressor = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            #nn.Tanh()  # Output in [-1, 1] range
        )
    
    def forward(self, input_ids, attention_mask, task_type='classification'):
        """
        Args:
            task_type: 'classification' or 'regression'
        """
        # Get encoder output
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        if task_type == 'classification':
            return self.classifier(pooled_output)
        else:  # regression
            return self.regressor(pooled_output).squeeze(-1)

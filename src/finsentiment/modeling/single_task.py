"""Model architecture."""

import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from finsentiment.config import MODEL_NAME

class FinancialSentimentModel(nn.Module):
    def __init__(self, model_name=None, num_classes=3):
        super().__init__()
        if model_name is None:
            model_name = MODEL_NAME
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            use_safetensors=True
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

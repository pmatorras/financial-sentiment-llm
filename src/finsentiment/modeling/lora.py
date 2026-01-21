import torch.nn as nn
from transformers import AutoModel
from peft import LoraConfig, get_peft_model, TaskType

class LoRAFinancialSentimentModel(nn.Module):
    def __init__(self, model_name, num_classes=3, lora_config=None):
        super().__init__()
        
        # Load Base Encoder
        self.encoder = AutoModel.from_pretrained(model_name, use_safetensors=True)
        
        # Apply LoRA Adapter
        if lora_config:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=lora_config['r'],
                lora_alpha=lora_config['alpha'],
                lora_dropout=lora_config['dropout'],
                target_modules=lora_config['target_modules']
            )
            self.encoder = get_peft_model(self.encoder, peft_config)
            self.encoder.print_trainable_parameters()
        
        # Task Heads (Standard)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes)
        )
        self.regressor = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask, task_type='classification'):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # PEFT models output a tuple where the first element is usually hidden states
        # but sometimes it wraps it differently. Safe access:
        if hasattr(outputs, 'last_hidden_state'):
            pooled_output = outputs.last_hidden_state[:, 0, :]
        else:
            pooled_output = outputs[0][:, 0, :]
            
        if task_type == 'classification':
            return self.classifier(pooled_output)
        return self.regressor(pooled_output).squeeze(-1)

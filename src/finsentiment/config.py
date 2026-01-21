from pathlib import Path
import random
import numpy as np
import torch

# Project root
ROOT_DIR = Path(__file__).resolve().parents[2]
# Data paths
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model paths
MODELS_DIR = ROOT_DIR / "models"

# Dataset URLs
PHRASEBANK_URL = "https://huggingface.co/datasets/takala/financial_phrasebank"
TWITTER_URL = "..."
FIQA_URL = "..."

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
MAX_LENGTH = 128
PATIENCE = 2

NUM_CLASSES = 3
CLAS_WEIGHT = 1.0
REGR_WEIGHT = 10.0

LORA_DEFAULT = {
    "r": 8,
    "alpha": 16,
    "dropout": 0.1,
    "target_modules": ["query", "value"]
}

LORA_TUNED = {
    "r": 16,           
    "alpha": 32,       
    "dropout": 0.1,
    "target_modules": ["query", "key", "value", "dense"]
}
# Reproducibility
SEED = 42

# Default to False: Experiments showed aggressive cleaning (removing short text)
# artificially inflated metrics on subsets but did not improve generalization
# on the full test set.
CLEAN_DATA_DEFAULT = False 

MODEL_REGISTRY = {
    'finbert': {
        'base_model': 'ProsusAI/finbert',
        'lora_config': None
    },
    'bert': {
        'base_model': 'bert-base-uncased',
        'lora_config': None
    },
    'distilbert': {
        'base_model': 'distilbert-base-uncased',
        'lora_config': None
    },
    'lora': {
        'base_model': 'ProsusAI/finbert',
        'lora_config': LORA_DEFAULT
    },
    'lora-tuned': {
        'base_model': 'ProsusAI/finbert',
        'lora_config': LORA_DEFAULT
    },
    'lora-r64': {
        'base_model': 'ProsusAI/finbert',
        'lora_config': {
            "r": 64,
            "alpha": 128,          # Maintain 2x ratio
            "dropout": 0.1,
            "target_modules": ["query", "key", "value", "dense"]
        }
    } 
}

def set_seed(seed=SEED):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CUDA deterministic (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model_path(model_name='base', model_type='single'):
    """Get default model checkpoint path."""
    return MODELS_DIR / f"{model_name}_{model_type}_task_model.pt"


def get_model_config(model_key: str) -> str:
    """
    Map a short CLI model key to the full HuggingFace model identifier.
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_key}' not found in registry. Available: {list(MODEL_REGISTRY.keys())}")
        
    return MODEL_REGISTRY[model_key]
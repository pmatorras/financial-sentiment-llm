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

# Model config
#MODEL_NAME = 'bert-base-uncased'
MODEL_NAME = 'ProsusAI/finbert'

NUM_CLASSES = 3

# Dataset weights
DATASET_WEIGHTS = {
    'phrasebank': 0.6,
    'twitter': 0.15,
    'fiqa': 0.25
}
DATASET_WEIGHTS_ALTERNATIVE = {
    'phrasebank': 0.4,#0.6,
    'twitter': 0.1,#0.15,
    'fiqa': 0.5,#0.25
}

# Reproducibility
SEED = 42

# Default to False: Experiments showed aggressive cleaning (removing short text)
# artificially inflated metrics on subsets but did not improve generalization
# on the full test set.
CLEAN_DATA_DEFAULT = False 

MODEL_REGISTRY = {
    'finbert': 'ProsusAI/finbert',
    'bert': 'bert-base-uncased',
    'distilbert': 'distilbert-base-uncased',
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

def resolve_model_name(model_key: str) -> str:
    """
    Map a short CLI model key to the full HuggingFace model identifier.
    Falls back to default if key is not found.
    """
    if model_key not in MODEL_REGISTRY:
        # Optional: warn user or raise error
        print(f"Warning: Model '{model_key}' not found in registry.")
        
    return MODEL_REGISTRY[model_key]
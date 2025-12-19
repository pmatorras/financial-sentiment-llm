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
MODEL_PATH = MODELS_DIR / f'sentiment_model_stop_testFiQA_{MODEL_NAME.split('/')[-1]}.pt'

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

def set_seed(seed=SEED):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CUDA deterministic (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

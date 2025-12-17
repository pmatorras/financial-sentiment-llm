from pathlib import Path

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
NUM_EPOCHS = 5
MAX_LENGTH = 128

# Model config
MODEL_NAME = "bert-base-uncased"
NUM_CLASSES = 3

# Dataset weights
DATASET_WEIGHTS = {
    'phrasebank': 0.6,
    'twitter': 0.15,
    'fiqa': 0.25
}

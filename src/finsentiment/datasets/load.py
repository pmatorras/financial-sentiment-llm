
import pandas as pd
from finsentiment.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from datasets import load_dataset
from finsentiment.datasets.clean_data import clean_dataset


def load_any_dataset(dataset_name='twitter', dataset_path='zeroshot/twitter-financial-news-sentiment', task_type = 'classification'):
    """Load A dataset given by an argument."""
    raw_path = RAW_DATA_DIR / f"{dataset_name}.csv"
    clean_path = PROCESSED_DATA_DIR / f"{dataset_name}_clean.csv"    
    # Download raw if needed
    if not raw_path.exists():
        print(f"Downloading {dataset_name} to {raw_path}...")
        dataset = load_dataset(dataset_path)
        df = pd.DataFrame(dataset['train'])
        if 'sentence' in df.columns:
            df = df.rename(columns={'sentence': 'text'})
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_path, index=False)
        print(f"✓ Saved raw data")
    
    # Clean if processed version doesn't exist
    if not clean_path.exists():
        print(f"Cleaning {dataset_name}...")
        df = pd.read_csv(raw_path)
        df = clean_dataset(dataset_name)
    
    # Load from processed/
    df = pd.read_csv(raw_path)

    if 'score' in df.columns: 
        df['label'] = df['score'].apply(lambda x: 0 if x < -0.1 else (2 if x > 0.1 else 1))
    else:
        df['score'] = 0.0
    df['task_type'] = task_type
    df['source'] = dataset_name
    return df

def load_fiqa(multi_task=False):
    """Load FiQA dataset with continuous scores."""
    raw_path = RAW_DATA_DIR / "fiqa.csv"
    clean_path = PROCESSED_DATA_DIR / "fiqa.csv"

    if raw_path.exists():
        df = pd.read_csv(raw_path)
    else:
        print(f"Downloading FiQA dataset to {raw_path}...")
        dataset = load_dataset("TheFinAI/fiqa-sentiment-classification")
        df = pd.DataFrame(dataset['train'])
        if 'sentence' in df.columns:
            df = df.rename(columns={'sentence': 'text'})
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_path, index=False)
        print(f"✓ Saved to {raw_path}")
    
    if multi_task:
        df['task_type'] = 'regression'
        if 'score' in df.columns:
            df['label'] = df['score'].apply(lambda x: 0 if x < -0.1 else (2 if x > 0.1 else 1))
    else:
        df['task_type'] = 'classification'
        # Convert continuous score to discrete label
        if 'score' in df.columns:
            def get_label(x):
                    if x < -0.1: return 0
                    if x > 0.1: return 2
                    return 1
                
            df['label'] = df['score'].apply(get_label)
    
    df['source'] = 'fiqa'
    return df

import pandas as pd
from finsentiment.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from datasets import load_dataset
from finsentiment.datasets.clean_data import clean_dataset


def load_any_dataset(dataset_name='twitter', dataset_path='zeroshot/twitter-financial-news-sentiment', task_type = 'classification', clean_data=True):
    """Load A dataset given by an argument."""
    raw_path = RAW_DATA_DIR / f"{dataset_name}.csv"
    clean_path = PROCESSED_DATA_DIR / f"{dataset_name}_clean.csv"
    output_path = clean_path if clean_data else raw_path   
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
    if not clean_path.exists() and clean_data:
        print(f"Cleaning {dataset_name}...")
        df = pd.read_csv(raw_path)
        df = clean_dataset(dataset_name)
    
    # Load from processed/
    df = pd.read_csv(output_path)

    if 'score' in df.columns: 
        df['label'] = df['score'].apply(lambda x: 0 if x < -0.1 else (2 if x > 0.1 else 1))
    else:
        df['score'] = 0.0
    df['task_type'] = task_type
    df['source'] = dataset_name
    return df
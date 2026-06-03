import kagglehub
import pandas as pd

def dataset(handle: str) -> None:
    try:
        dataset_dir = kagglehub.dataset_download(handle) 
        print(f'Dataset {handle} downloaded to {dataset_dir}')
    except ValueError:
        dataset_dir = kagglehub.competition_download(handle)
        print(f'Dataset {handle} downloaded to {dataset_dir}')
    except Exception as e:
        print(f'Error downloading dataset: {e}')

def load_train_df(cfg):
    return pd.read_csv(cfg.dataset.train_path)

def load_test_df(cfg):
    return pd.read_csv(cfg.dataset.test_path)
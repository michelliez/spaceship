import hydra
import pandas as pd

from utils.train_utils import dataset, load_test_df, load_train_df

@hydra.main(version_base=None, config_path="../../config/hyp1", config_name='config_1')
def train(cfg):
    train_df=load_train_df(cfg)
    #df=pd.read_csv(cfg.dataset.train_path)
    print(train_df.describe())
    print(train_df.head())

if __name__ == "__main__":
    train()

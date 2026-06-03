#Hypothesis 1: HomePlanet, CryoSleep, Cabin, Destination, Age, VIP affect transportation.
#Drop data with missing values.

import hydra
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from utils.train_utils import dataset, load_test_df, load_train_df

@hydra.main(version_base=None, config_path="../../config/hyp1", config_name='config_1')
def train(cfg):

    #Pandas DF
    train_df=load_train_df(cfg)

    #Prediction target: Transported or not
    y = train_df.Transported

    #Features (as given by hypothesis): HomePlanet, CryoSleep, Cabin, Destination, Age, VIP
    train_features = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP']
    
    #drop row if it is missing data in desired columns
    train_df = train_df.dropna(subset=train_features)
    print(train_df.describe())

    X = train_df[train_features]
    print(X.describe())
    print(X.head())

    #Split training data into train and valid
    # train_X, val_X, train_y, val_y
    # cols_with_missing = [col for col in train_df.columns if train_df[col].isnull().any()]
    # reduced_







if __name__ == "__main__":
    train()

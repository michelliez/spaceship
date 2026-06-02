import os
from dotenv import load_dotenv

load_dotenv()
print(os.environ['KAGGLE_USERNAME'])
print(os.environ['KAGGLE_KEY'])


import sys
import kagglehub

dataset_name = sys.argv[1]
dataset = kagglehub.competition_download(dataset_name) 
    #spaceship-titanic

# print (dataset)
# print(os.listdir(dataset))


import pandas as pd
train_file = dataset + '/train.csv'
test_file = dataset + '/test.csv'
train_df = pd.read_csv(train_file)   
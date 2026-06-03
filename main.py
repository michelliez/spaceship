# import os
# from dotenv import load_dotenv

# load_dotenv()
# print(os.environ['KAGGLE_USERNAME'])
# print(os.environ['KAGGLE_KEY'])

#def get_data_frames(dataset_name):

import sys
import kagglehub
import pandas as pd

dataset_name = sys.argv[1]

dataset_name = sys.argv[1]
dataset = kagglehub.competition_download(dataset_name) 
    #spaceship-titanic

# print (dataset)
# print(os.listdir(dataset))

train_file = dataset + '/train.csv'
test_file = dataset + '/test.csv'
train_df = pd.read_csv(train_file)   
test_df = pd.read_csv(test_file)    
print(train_df.head())

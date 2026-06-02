import os
from dotenv import load_dotenv

load_dotenv()
print(os.environ['KAGGLE_USERNAME'])
print(os.environ['KAGGLE_KEY'])


import sys
import kagglehub

#dataset_name = sys.argv[1]
#path_name = sys.argv[2]

dataset = kagglehub.competition_download('spaceship-titanic') 
    #competitions/spaceship-titanic train.csv
    
# print (dataset)
# print(os.listdir(dataset))


import pandas as pd
train_file = dataset + '/train.csv'
test_file = dataset + '/test.csv'
train_df = pd.read_csv(train_file)   
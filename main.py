import os
from dotenv import load_dotenv

load_dotenv()
print(os.environ['KAGGLE_USERNAME'])


import sys
import kagglehub

dataset_name = sys.argv[1]
dataset = kagglehub.competition_download('dataset_name', path='./data', unzip = True) #competitions/spaceship-titanic


import pandas as pd
df = pd.read_csv(dataset)
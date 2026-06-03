# Spaceship Titanic
This repo contains my solution for the **Spaceship Titanic** Kaggle competition.  

# 1. Setup

## 1.1 Kaggle API
Please replace the placeholder Kaggle username and API with your own, as follows in bash:

``` bash
export KAGGLE_USERNAME="your_username"; 
export KAGGLE_KEY="your_api"
```

## 1.2 Downloading Data
Data was downloaded via kagglehub from spaceship-titanic. My code is shown in *dataset.py*. Run `python dataset.py` to run the script that downloads the competition dataset (spaceship-titanic).  

# 2. Training
## 2.1 Pandas DataFrames 
A pandas DataFrame was created for **train.csv** in `train.py`, which first returns a description of the DataFrame, and the first few rows. Run with `uv run -m hyp1.train` inside the **code** folder.  
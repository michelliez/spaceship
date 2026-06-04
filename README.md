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

## 1.3 EDA
EDA was conducted in a Jupyter notebook under **notebooks**, **eda1.ipynb**. The resulting findings are:
- There are almost 2x more training data than testing data.
- Data regarding transportation status was collected for all training data.
- Most of the data was continuous.
- The continuous data had similar distributions for both training and testing data.
- Most passengers were young adults (youth).
- There was a good split between Transported and Not Transported. 
- More passengers whose ages range from 0-<20 were transported.
- Less adults, age range 20-40, were transported.
Overall, fewer people were frozen.
- The proportion of frozen in transported passengers was higher. This could mean that those frozen were Transported at a slightly higher rate than nonfrozen. 
- Overall distribution of transportation split by cabins was similar throughout, except for cabin G.
- The number of non transported passengers was greater in cabin G.
- For groups of more than 1, if one member of the group was transported, there is a higher chance that other members of the group was also transported.
- Passengers from Europa had more transports.
- Passengers to TRAPPIST-1e MAY have fewer transports.
- 55 Cancri e has higher transportation rates.
- People who don't purchase premium add-ons were transported at higher rates.
Overall, transportation could be selective impacted by how much a passenger spends, which may also explain why people who are frozen in CryoSleep are transported at higher rates. Those in similar spending ranges also move around the ship in close proximity, which could explain Destination/HomePlanet and cabin patterns.


# 2. Training
## 2.1 Pandas DataFrames 
A pandas DataFrame was created for **train.csv** in `train.py`, which first returns a description of the DataFrame, and the first few rows. Run with `uv run -m hyp1.train` inside the **code** folder.  
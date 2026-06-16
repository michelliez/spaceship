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
- Overall, fewer people were frozen.
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
## 2.1 DataFrames & Feature Engineering
Pandas DataFrames were created for **train.csv** and **test.csv** in `train.py`.   

## 2.2 Feature Engineering
Additional features were added to the DataFrame based on features in `preprocessing.py`. Such features include:
- Cabin Deck
- Cabin Side
- Cabin Number
- Cabin Missing (Boolean)  
- Passenger Group (First 4 numbers in PassengerID)
- Group Number
- Group Size
- Solo Travel (Boolean)
- Last Name
- Family Size
- Total Spending
- Socioeconomic Status
- Luxury Spending
- Essential Spending
- No Spend
- Cryo No Spend

## 2.3 Imputation
Based on EDA, missing data was imputed.
- No one from Earth was a VIP, therefore imputed with False.
- No one under 18 was a VIP, therefore imputed with False.
- Passengers under 13 did not spend any money, therefore imputed with 0.
- Passengers in CryoSleep did not spend any money, therefore imputed with 0.
- Assuming that groups are families, they: 1. Have the same last name. 2: Have same destination and home planet. 3: Live in the same cabin deck and side.
- Anymore who spends money are not in CryoSleep, therefore imputed with False.
- Remaining missing numerical values were imputed with the column's median, and missing categorical values were imputed with the mode (or left as 'Missing' for CatBoost)

## 2.4 Baseline Logistic Regression Classification Model
A baseline model was created `code/hyp1` with `code/hyp1/preprocessing.py`, `code/hyp1/model.py`, and `code/hyp1/train.py`. This model used basic feature engineering. A **Logistic Regression** was used to predict transportation status. The train.csv pandas DataFrame was split into 80% training data and 20% validation data using sklearn train-test-split.   
Run with `uv run -m hyp1.train` inside the **code** directory.  

## 2.5 Decision Trees - XGBoost
Out of **RandomForestClassifier**, **GradientBoostingClassifier**, **HistGradientBoostingClassifier**, and **XGBoostClassifier**, **XGBoostClassifier** yielded the highest accuracy. Features were improved in `code/hyp2/preprocessing.py`. See Section **2.2**. The **RepeatedStratifiedKFold** and **train_test_split** cross-validiation method was used (RepeatedStratifiedKFold is currently commented out. Please comment out sklearn's train_test_split and uncomment KFold to use). Additionally, data was imputed via the strategy mentioned in Section **2.3**, and **normalized** using StandardScaler.  
Run with `uv run -m hyp2.train` inside the **code** directory.  
Results were submitted to the **Spaceship Titanic** Kaggle competition with an accuracy of **0.80547**.

## 2.6 Neural Network
A simple **Torch** neural network found in `code/hyp3` with activation function ReLU resulted in a lower Kaggle accuracy of **0.79635**. However, data was not embedded, likely resulting in a lower score than decision tree models. Further, neural networks are weaker for tabular data such as **Spaceship Titanic**, which decision trees can easily learn.  
Run with `uv run -m hyp3.train` inside the **code** directory.  

## 2.7 Decision Trees Cont. - CatBoost
The CatBoostClassifier was chosen due to its permutation-driven algorithm to encode categorical features. Instead of imputing missing categorical values with the modes, missing values were left as 'Missing' to allow the model to encode for missing information. This model yielded the highest Kaggle competition accuracy at **0.80687** resulting in a rank of **602/2198**.
Run with `uv run -m hyp4.train` inside the **code** directory.  

# 3. Kaggle Submission
Results were submitted to the **Spaceship Titanic** Kaggle competition.
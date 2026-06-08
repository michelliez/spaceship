import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

features = ['Age', 'CryoSleep', 'CabinDeck', 'PassengerGroup', 'GroupNumber', 'Destination', 'HomePlanet', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'FamilySize', 'VIP']
cont_cols=['Age', 'PassengerGroup', 'GroupNumber', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'FamilySize']
cat_cols=['Destination', 'HomePlanet', 'CryoSleep', 'VIP', 'CabinDeck']

def add_features (train_df, test_df):
    train_df['CabinDeck'] = train_df['Cabin'].str[0]
    train_df['PassengerGroup'] = train_df.PassengerId.str.split('_').str[0].astype(int) 
    train_df['GroupNumber'] = train_df.PassengerId.str.split('_').str[1].astype(int)

    test_df['CabinDeck'] = test_df['Cabin'].str[0]
    test_df['PassengerGroup'] = test_df.PassengerId.str.split('_').str[0].astype(int)
    test_df['GroupNumber'] = test_df.PassengerId.str.split('_').str[1].astype(int)

    train_df['LastName'] = train_df.Name.str.split(' ').str[-1]
    test_df['LastName'] = test_df.Name.str.split(' ').str[-1]
    train_df['FamilySize'] = train_df['LastName'].map(train_df['LastName'].value_counts())
    test_df['FamilySize'] = test_df['LastName'].map(test_df['LastName'].value_counts())
    return train_df, test_df

def get_preprocessor():
    cont_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy = 'most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('cont', cont_pipeline, cont_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    return preprocessor
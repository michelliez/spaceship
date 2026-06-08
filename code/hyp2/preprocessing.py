import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

features = ['Age', 'CryoSleep', 'CabinDeck', 'CabinSide', 'CabinNumber', 'PassengerGroup', 'TotalSpending', 'SocioEconStatus', 'NoSpend', 'LuxurySpend','CryoNoSpend', 'Destination', 'HomePlanet', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'VIP']
cont_cols=['Age', 'CabinNumber', 'PassengerGroup', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpending', 'NoSpend', 'LuxurySpend', 'CryoNoSpend']
cat_cols=['Destination', 'HomePlanet', 'CryoSleep', 'VIP', 'CabinDeck', 'CabinSide', 'SocioEconStatus']

def add_features (train_df, test_df):
    train_df['CabinDeck'] = train_df['Cabin'].str[0]
    train_df['CabinSide'] = train_df['Cabin'].str[2]
    train_df['CabinNumber'] = pd.to_numeric(train_df['Cabin'].str.split('/').str[1])
    train_df['CabinMissing'] = train_df['Cabin'].isna().astype(int)
    train_df['PassengerGroup'] = train_df.PassengerId.str.split('_').str[0].astype(int) 
    # train_df['GroupNumber'] = train_df.PassengerId.str.split('_').str[1].astype(int)
    train_df['TotalSpending'] = (train_df['RoomService'] + train_df['FoodCourt'] + train_df['ShoppingMall'] + train_df['Spa'] + train_df['VRDeck'])
    train_df["SocioEconStatus"] = train_df["TotalSpending"].apply(lambda x: "Low" if x < 5000 else ("Middle" if x>5000 and x<20000 else "Upper"))
    train_df['LuxurySpend'] = train_df['Spa'] + train_df['VRDeck'] + train_df['RoomService']
    train_df['NoSpend'] = (train_df['TotalSpending']==0).astype(int)
    train_df['CryoNoSpend'] = ((train_df['CryoSleep']==True) & train_df['NoSpend']==True).astype(int)
    # train_df['GroupSize'] = (train_df['PassengerGroup'].map(train_df['PassengerGroup'].value_counts()))
    # train_df['SoloTravel'] = (train_df['GroupSize']==1).astype(int)

    test_df['CabinDeck'] = test_df['Cabin'].str[0]
    test_df['CabinSide'] = test_df['Cabin'].str[2]    
    test_df['CabinNumber'] = pd.to_numeric(test_df['Cabin'].str.split('/').str[1])
    test_df['CabinMissing'] = test_df['Cabin'].isna().astype(int)
    test_df['PassengerGroup'] = test_df.PassengerId.str.split('_').str[0].astype(int)
    # test_df['GroupNumber'] = test_df.PassengerId.str.split('_').str[1].astype(int)
    test_df['TotalSpending'] = (test_df['RoomService'] + test_df['FoodCourt'] + test_df['ShoppingMall'] + test_df['Spa'] + test_df['VRDeck'])
    test_df["SocioEconStatus"] = test_df["TotalSpending"].apply(lambda x: "Low" if x < 5000 else ("Middle" if x>5000 and x<20000 else "Upper"))
    test_df['LuxurySpend'] = test_df['Spa'] + test_df['VRDeck'] + test_df['RoomService']
    test_df['NoSpend'] = (test_df['TotalSpending']==0).astype(int)
    test_df['CryoNoSpend'] = ((test_df['CryoSleep']==True) & test_df['NoSpend']==True).astype(int)
    # test_df['GroupSize'] = (test_df['PassengerGroup'].map(test_df['PassengerGroup'].value_counts()))
    # test_df['SoloTravel'] = (test_df['GroupSize']==1).astype(int)

    
    # train_df['LastName'] = train_df.Name.str.split(' ').str[-1]
    # test_df['LastName'] = test_df.Name.str.split(' ').str[-1]
    # train_df['FamilySize'] = train_df['LastName'].map(train_df['LastName'].value_counts())
    # test_df['FamilySize'] = test_df['LastName'].map(test_df['LastName'].value_counts())

    return train_df, test_df

def get_preprocessor():
    cont_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        # ('scaler', StandardScaler())
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
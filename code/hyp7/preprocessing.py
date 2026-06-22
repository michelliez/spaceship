import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, Normalizer
from feature_engine.outliers import Winsorizer
from sklearn.impute import SimpleImputer

features = ['Age', 'CryoSleep', 'CabinDeck', 'CabinSide', 'CabinNumber', 'GroupSize', 'FamilySize', 'SoloTravel', 'TotalSpending', 'NoSpend', 'LuxurySpend', 'EssentialSpend', 'Destination', 'HomePlanet', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'VIP']
cont_cols=['Age', 'CabinNumber', 'GroupSize', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpending',  'LuxurySpend', 'EssentialSpend', 'FamilySize']
cat_cols=['Destination', 'HomePlanet', 'CryoSleep', 'VIP', 'CabinDeck', 'CabinSide', 'NoSpend', 'SoloTravel']

def engineer_features (train_df, test_df):
    # full = pd.concat([train_df, test_df], ignore_index=True)
    # #Group / ID
    # group = full['PassengerId'].str.split('_', expand = True)
    # full['PassengerGroup'] = pd.to_numeric(group[0], errors = 'coerce')
    # full['GroupNumber'] = pd.to_numeric(group[1], errors = 'coerce')
    # full['GroupSize'] = full.groupby('GroupId')['PassengerId'].transform('size').astype(int)
    # full['Solo'] = (full['GroupSize']==1).astype(int)

    # #Cabin
    # full['CabinDeck'] = full['Cabin'].str.split('/').str[0]
    # full['CabinNumber'] = pd.to_numeric(full['Cabin'].str.split('/').str[1])
    # full['CabinSide'] = full['Cabin'].str.split('/').str[2]
    # full['CabinNumParity'] = (full['CabinNumber']%2).fillna(-1).astype(int)
    # full['CabinNumBucket'] = pd.cut(full['CabinNumber'], bins=10, labels=False, duplicates='drop').fillna(-1).astype(int)

    # #Name / Family
    # full['LastName'] = full.Name.str.split(' ').str[-1].fillna('Unknown')
    # full['FamilySize'] = full['LastName'].map(full['LastName'].value_counts())
    # full['SurnameGroupSize'] = full.groupby('LastName')['PassengerGroup'].transform('nunique').astype(int)
    
    # #Infer CryoSleep via Spending
    # spend0 = full['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'].fillna(0).sum(axis=1)
    # full.loc[full['CryoSleep'].isna() & (sp > 0), 'CryoSleep'] = False
    # full.loc[full['CryoSleep'].isna() & (sp == 0), 'CryoSleep'] = True

    train_df['CabinDeck'] = train_df['Cabin'].str.split('/').str[0]
    train_df['CabinNumParity'] = (train_df['CabinNumber']%2).fillna(-1).astype(int)
    train_df['CabinNumBucket'] = pd.cut(train_df['CabinNumber'], bins=10, labels=False, duplicates='drop').fillna(-1).astype(int)
    train_df['CabinSide'] = train_df['Cabin'].str.split('/').str[2]
    train_df['CabinNumber'] = pd.to_numeric(train_df['Cabin'].str.split('/').str[1])
    train_df['CabinMissing'] = train_df['Cabin'].isna().astype(int)
    train_df['PassengerGroup'] = train_df.PassengerId.str.split('_').str[0].astype(int) 
    train_df['GroupNumber'] = train_df.PassengerId.str.split('_').str[1].astype(int)
    train_df['GroupSize'] = (train_df['PassengerGroup'].map(train_df['PassengerGroup'].value_counts()))
    train_df['SoloTravel'] = (train_df['GroupSize']==1).astype(int)
    train_df['LastName'] = train_df.Name.str.split(' ').str[-1]
    train_df['FamilySize'] = train_df['LastName'].map(train_df['LastName'].value_counts())
    train_df['AgeGroup'] = pd.cut(train_df['Age'], bins=[-1, 5, 13, 18, 60, 100], labels=['Baby', 'Child', 'Teen', 'Adult', 'Elderly'])

    test_df['CabinDeck'] = test_df['Cabin'].str.split('/').str[0]
    test_df['CabinNumParity'] = (test_df['CabinNumber']%2).fillna(-1).astype(int)
    test_df['CabinNumBucket'] = pd.cut(test_df['CabinNumber'], bins=10, labels=False, duplicates='drop').fillna(-1).astype(int)
    test_df['CabinSide'] = test_df['Cabin'].str.split('/').str[2]    
    test_df['CabinNumber'] = pd.to_numeric(test_df['Cabin'].str.split('/').str[1])
    test_df['CabinMissing'] = test_df['Cabin'].isna().astype(int)
    test_df['PassengerGroup'] = test_df.PassengerId.str.split('_').str[0].astype(int)
    test_df['GroupNumber'] = test_df.PassengerId.str.split('_').str[1].astype(int)
    test_df['GroupSize'] = (test_df['PassengerGroup'].map(test_df['PassengerGroup'].value_counts()))
    test_df['SoloTravel'] = (test_df['GroupSize']==1).astype(int)
    test_df['LastName'] = test_df.Name.str.split(' ').str[-1]
    test_df['FamilySize'] = test_df['LastName'].map(test_df['LastName'].value_counts())
    test_df['AgeGroup'] = pd.cut(test_df['Age'], bins=[-1, 5, 13, 18, 60, 100], labels=['Baby', 'Child', 'Teen', 'Adult', 'Elderly'])


    return train_df, test_df

def predict_missing_values(data):
    #No VIP in Earth; No VIP < 18
    data.loc[data['VIP'].isna() & (data['HomePlanet'] == 'Earth'), 'VIP'] = False
    data.loc[data['VIP'].isna() & (data['Age'] < 18), 'VIP'] = False

    #Passengers < 13 and Passengers Sleeping have no spending
    spending_cols = ['Spa', 'ShoppingMall', 'VRDeck', 'FoodCourt', 'RoomService']
    for col in spending_cols:
        data.loc[data[col].isna() & ((data['Age'] < 13 ) | (data['CryoSleep'] == True)), col] = 0

    #Assume groups are families, and have the same last name, and have same destination and home planet, and same cabin deck+side
    group_cols = ['LastName', 'HomePlanet', 'Destination', 'CabinDeck', 'CabinSide']
    for col in group_cols:
        data[col] = data.groupby('PassengerGroup')[col].transform( lambda x: x.ffill().bfill())

    #Anyone who spends is not asleep
    data.loc[data['CryoSleep'].isna() & ((data['Spa']>0) | (data['RoomService']>0) | (data['FoodCourt']>0) | (data['VRDeck']>0) | (data['ShoppingMall']>0)), 'CryoSleep'] = False
       
    return data

def add_spending (train_df, test_df):
    train_df['TotalSpending'] = (train_df['RoomService'] + train_df['FoodCourt'] + train_df['ShoppingMall'] + train_df['Spa'] + train_df['VRDeck'])
    train_df["SocioEconStatus"] = train_df["TotalSpending"].apply(lambda x: "Low" if x < 5000 else ("Middle" if x < 20000 else "Upper"))
    train_df['LuxurySpend'] = train_df['Spa'] + train_df['VRDeck'] + train_df['RoomService']
    train_df['EssentialSpend'] = train_df['FoodCourt'] + train_df['ShoppingMall']
    train_df['NoSpend'] = (train_df['TotalSpending']==0).astype(int)
    train_df['CryoNoSpend'] = ((train_df['CryoSleep'] == True) & (train_df['NoSpend'] == 1)).astype(int)

    test_df['TotalSpending'] = (test_df['RoomService'] + test_df['FoodCourt'] + test_df['ShoppingMall'] + test_df['Spa'] + test_df['VRDeck'])
    test_df["SocioEconStatus"] = test_df["TotalSpending"].apply(lambda x: "Low" if x < 5000 else ("Middle" if x < 20000 else "Upper"))
    test_df['LuxurySpend'] = test_df['Spa'] + test_df['VRDeck'] + test_df['RoomService']
    test_df['EssentialSpend'] = test_df['FoodCourt'] + test_df['ShoppingMall']
    test_df['NoSpend'] = (test_df['TotalSpending']==0).astype(int)
    test_df['CryoNoSpend'] = ((test_df['CryoSleep'] == True) & (test_df['NoSpend'] == 1)).astype(int)

    return train_df, test_df

def get_cat_preprocessor(X_train, X_valid, X_test):
    for col in cont_cols:
        median = X_train[col].median()
        X_train[col] = X_train[col].fillna(median)
        X_valid[col] = X_valid[col].fillna(median)
        X_test[col] = X_test[col].fillna(median)


    for col in cat_cols:
        X_train[col] = X_train[col].astype(str).fillna('Missing')
        X_valid[col] = X_valid[col].astype(str).fillna('Missing')
        X_test[col] = X_test[col].astype(str).fillna('Missing')

    return X_train, X_valid, X_test

def get_preprocessor():
    cont_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        # ('winsor', Winsorizer(capping_method='iqr', tail='both', fold=1.5)),
        ('normalizer', StandardScaler())
        # ('normalizer', Normalizer())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy = 'most_frequent')),
        ('encoder', OrdinalEncoder())
    ])
    preprocessor = ColumnTransformer([
        ('cont', cont_pipeline, cont_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    return preprocessor
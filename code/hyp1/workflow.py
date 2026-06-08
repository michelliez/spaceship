#Features: Age, CryoSleep, CabinDeck, PassengerGroup, Destination, HomePlanet, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck, VIP
#Model1: 80-20 train test split

import hydra
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

from utils.train_utils import dataset, load_test_df, load_train_df

@hydra.main(version_base=None, config_path="../../config/hyp1", config_name='config_1')
def workflow(cfg):

    #Pandas DF
    train_df=load_train_df(cfg)
    test_df=load_test_df(cfg)

    #Prediction target: Transported or not
    y = train_df.Transported.astype(int)

    #Extra Features
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
 
    #Features: Age, CryoSleep, CabinDeck, PassengerGroup, GroupNumber, Destination, HomePlanet, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck, FamilySize, VIP
    features = ['Age', 'CryoSleep', 'CabinDeck', 'PassengerGroup', 'GroupNumber', 'Destination', 'HomePlanet', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'FamilySize', 'VIP']
    X = train_df[features]

    #Split training data into train and valid
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=23)
    
    #Imputation: Continuous with median, Categorical with most common.
    cont_cols=['Age', 'PassengerGroup', 'GroupNumber', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'FamilySize']
    X_train[cont_cols] = X_train[cont_cols].fillna(X_train[cont_cols].median())
    X_valid[cont_cols] = X_valid[cont_cols].fillna(X_valid[cont_cols].median())

    cat_cols=['Destination', 'HomePlanet', 'CryoSleep', 'VIP', 'CabinDeck']
    imputer = SimpleImputer(strategy='most_frequent')
    X_train[cat_cols] = imputer.fit_transform(X_train[cat_cols])
    X_valid[cat_cols] = imputer.transform(X_valid[cat_cols])

    #Convert CryoSleep and VIP Booleans to ints
    X_train['CryoSleep'] = X_train['CryoSleep'].astype(int)
    X_train['VIP'] = X_train['VIP'].astype(int)

    X_valid['CryoSleep'] = X_valid['CryoSleep'].astype(int)
    X_valid['VIP'] = X_valid['VIP'].astype(int)

    #Encode remaining categorical vars with OneHotEncoder
    enc_columns = ['CabinDeck', 'Destination', 'HomePlanet']
    enc = OneHotEncoder(sparse_output=False)
    
    train_enc_data = enc.fit_transform(X_train[enc_columns])
    train_encoded_df = pd.DataFrame(
        train_enc_data,
        columns=enc.get_feature_names_out(enc_columns),
        index= X_train.index
    )
    X_train = pd.concat(
        [X_train.drop(columns=enc_columns), train_encoded_df],
        axis=1
    )

    valid_enc_data = enc.transform(X_valid[enc_columns])
    valid_encoded_df = pd.DataFrame(
        valid_enc_data,
        columns=enc.get_feature_names_out(enc_columns),
        index= X_valid.index
    )
    X_valid = pd.concat(
        [X_valid.drop(columns=enc_columns), valid_encoded_df],
        axis=1
    )

    #Scale Continuous Columns
    scaler = StandardScaler()
    X_train[cont_cols] = scaler.fit_transform(X_train[cont_cols])
    X_valid[cont_cols] = scaler.transform(X_valid[cont_cols])

    #Model
    model = LogisticRegression(max_iter=10000, random_state=0)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_valid)
    acc = accuracy_score(y_valid, y_hat)
    print('Accuracy on validation set: %.3f' %(acc*100))

    #Test on Testing data
    X_test = test_df[features]
    X_test[cont_cols] = X_test[cont_cols].fillna(X_train[cont_cols].median())
    X_test[cat_cols] = imputer.transform(X_test[cat_cols])
    test_enc_data = enc.transform(X_test[enc_columns])
    test_encoded_df = pd.DataFrame(
        test_enc_data,
        columns=enc.get_feature_names_out(enc_columns),
        index= X_test.index
    )
    X_test = pd.concat(
        [X_test.drop(columns=enc_columns), test_encoded_df],
        axis=1
    )
    X_test[cont_cols] = scaler.transform(X_test[cont_cols])

    #Predict on test data
    predict = model.predict(X_test)
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Transported': predict.astype(bool)
    })
    submission.to_csv('submission.csv', index=False)




if __name__ == "__main__":
    workflow()

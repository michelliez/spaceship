import hydra
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.train_utils import load_train_df, load_test_df
from hyp4.preprocessing import add_features, add_spending, features, predict_missing_values, get_preprocessor
from hyp4.model import build_model

@hydra.main(version_base=None, config_path="../../config/hyp4", config_name='config_4')
def train(cfg):
    #Pandas DF
    train_df=load_train_df(cfg)
    test_df=load_test_df(cfg)
    
    #Features
    train_df, test_df = add_features(train_df, test_df)

    #Manually fill in missing values
    train_df = predict_missing_values(train_df)
    test_df = predict_missing_values(test_df)
    
    #Add spending Features
    train_df, test_df = add_spending(train_df, test_df)

    #X and y
    X = train_df[features]
    y = train_df['Transported'].astype(int)
   
    #Split training data into train and valid
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=cfg.train_parameters.test_size, random_state=cfg.train_parameters.state)

    #Create test DF
    X_test = test_df[features]

    #Normalizing
    X_train, X_valid, X_test = get_preprocessor(X_train, X_valid, X_test)

    #Model
    model = build_model()
    cat_cols = ['Destination', 'HomePlanet', 'CryoSleep', 'VIP', 'CabinDeck', 'CabinSide', 'SocioEconStatus']
    model.fit(X_train, y_train, cat_features = cat_cols, eval_set=(X_valid, y_valid), use_best_model=True)
    y_hat = model.predict(X_valid)
    acc = accuracy_score(y_valid, y_hat)
    print('Accuracy on validation set: %.5f' %(acc*100))

    #Predict
    predict = model.predict(X_test)
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Transported': predict.astype(bool)
    })
    submission.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    train()


import hydra
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.train_utils import load_train_df, load_test_df
from hyp1.preprocessing import add_features, features
from hyp1.model import build_model

@hydra.main(version_base=None, config_path="../../config/hyp1", config_name='config_1')
def train(cfg):
    #Pandas DF
    train_df=load_train_df(cfg)
    test_df=load_test_df(cfg)
    
    #Features
    train_df, test_df = add_features(train_df, test_df)

    #X and y
    X = train_df[features]
    y = train_df['Transported'].astype(int)

    #Split training data into train and valid
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=cfg.train_parameters.test_size, random_state=cfg.train_parameters.state)

    #Model
    model = build_model()
    model.fit(X_train, y_train)

    #Validate
    y_hat = model.predict(X_valid)
    acc = accuracy_score(y_valid, y_hat)
    print('Accuracy on validation set: %.3f' %(acc*100))

    #Predict
    X_test = test_df[features]
    predict = model.predict(X_test)
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Transported': predict.astype(bool)
    })
    submission.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    train()


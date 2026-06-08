import hydra
import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from utils.train_utils import load_train_df, load_test_df
from hyp2.preprocessing import add_features, features
from hyp2.model import build_model

@hydra.main(version_base=None, config_path="../../config/hyp2", config_name='config_2')
def train(cfg):
    #Pandas DF
    train_df=load_train_df(cfg)
    test_df=load_test_df(cfg)
    
    #Features
    train_df, test_df = add_features(train_df, test_df)

    #X and y
    X = train_df[features]
    y = train_df['Transported'].astype(int)

    #CV
    cross_valid = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)

    #Model
    model = build_model()
    scores = cross_val_score(model, X, y, cv = cross_valid, scoring='accuracy')
    print(f"CV accuracy: {scores.mean() * 100:.3f}%")
    model.fit(X, y)

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


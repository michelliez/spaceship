import pandas as pd
import hydra
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from utils.train_utils import load_train_df, load_test_df
from sklearn.metrics import accuracy_score
from hyp6.preprocessing import add_features, add_spending, features, predict_missing_values, get_preprocessor
from hyp6.model import build_tabnet_model
import numpy as np
import warnings
warnings.filterwarnings(
    "ignore",
    message="Best weights from best epoch are automatically used!",
)
warnings.filterwarnings(
    "ignore",
    message= 'Device used'
)


@hydra.main(version_base=None, config_path="../../config/hyp4", config_name='config_4')
def train(cfg):
    train_df = load_train_df(cfg)
    test_df = load_test_df(cfg)

    train_df, test_df = add_features(train_df, test_df)

    train_df = predict_missing_values(train_df)
    test_df = predict_missing_values(test_df)

    train_df, test_df = add_spending(train_df, test_df)

    X = train_df[features]
    y = train_df['Transported'].astype(int)
    X_test = test_df[features]

    y=y.to_numpy()
    y=y.flatten()

    group_kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    test_probs = []
    valid_scores = []
    for fold, (train_idx, valid_idx) in enumerate(group_kf.split(X, y, groups=train_df['PassengerGroup'])):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        preprocessor = get_preprocessor()
        preprocessor.fit(X_train)
        X_train = preprocessor.transform(X_train)
        X_valid = preprocessor.transform(X_valid)
        X_test = preprocessor.transform(X_test)

        model = build_tabnet_model()
        model.fit(X_train, y_train,
                  eval_set = [(X_train, y_train), (X_valid, y_valid)],
                  eval_name = ['train', 'valid'],
                  eval_metric = ['accuracy'],
                  max_epochs=200,
                  patience=25,
                  batch_size=256,
                  virtual_batch_size=128,
                  num_workers=0,
                  drop_last=True)
        
        valid_prob = model.predict_proba(X_valid)[:, 1]
        valid_pred = valid_prob >= 0.5
        acc = accuracy_score(y_valid, valid_pred)
        valid_scores.append(acc)
        print('\nFold:', fold, 'Accuracy Score:', acc)

        test_prob = model.predict_proba(X_test)[:, 1]
        test_probs.append(test_prob)

    print(valid_scores)
    print('Mean CV Score:', np.mean(valid_scores), '\n')

    final_prob = np.mean(test_probs, axis=0)
    final_pred = final_prob >= 0.5
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Transported': final_pred
    })
    submission.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    train()
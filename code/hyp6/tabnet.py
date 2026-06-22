import pandas as pd
import hydra
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from utils.train_utils import load_train_df, load_test_df
from sklearn.metrics import accuracy_score
from hyp6.preprocessing import add_features, add_spending, features, predict_missing_values, get_preprocessor
from hyp6.model import build_tabnet_model
import numpy as np
from pathlib import Path

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
def get_tabnet_prob(cfg):
    train_df = load_train_df(cfg)
    test_df = load_test_df(cfg)

    train_df, test_df = add_features(train_df, test_df)

    train_df = predict_missing_values(train_df)
    test_df = predict_missing_values(test_df)

    train_df, test_df = add_spending(train_df, test_df)

    X = train_df[features]
    y = train_df['Transported'].astype(int).to_numpy().flatten()
    X_test = test_df[features]

    group_kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_probs = np.zeros(len(y))
    test_probs = []
    valid_scores = []

    for fold, (train_idx, valid_idx) in enumerate(group_kf.split(X, y, groups=train_df['PassengerGroup'])):
        X_train, X_valid = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
        X_test_fold = X_test.copy()
        y_train, y_valid = y[train_idx], y[valid_idx]

        preprocessor = get_preprocessor()
        preprocessor.fit(X_train)
        X_train = preprocessor.transform(X_train)
        X_valid = preprocessor.transform(X_valid)
        X_test_fold = preprocessor.transform(X_test_fold)

        model = build_tabnet_model()
        model.fit(X_train, y_train,
                  eval_set = [(X_valid, y_valid)],
                  eval_name = ['valid'],
                  eval_metric = ['accuracy'],
                  max_epochs=200,
                  patience=15,
                  batch_size=512,
                  virtual_batch_size=128,
                  num_workers=0,
                  drop_last=True)
        
        valid_prob = model.predict_proba(X_valid)[:, 1]
        oof_probs[valid_idx] = valid_prob
        valid_pred = valid_prob >= 0.5
        acc = accuracy_score(y_valid, valid_pred)
        valid_scores.append(acc)

        test_prob = model.predict_proba(X_test_fold)[:, 1]
        test_probs.append(test_prob)
    
    oof_pred = oof_probs >= 0.5
    oof_acc = accuracy_score(y, oof_pred)
    print('\nFold scores:', valid_scores)
    print('Mean Fold CV Score:', np.mean(valid_scores))
    print('OOF CV Score:', oof_acc)

    final_prob = np.mean(test_probs, axis=0)
    final_pred = final_prob >= 0.5

    save_dir = Path('saved_probs')
    save_dir.mkdir(exist_ok=True)
    np.save(save_dir / 'tab_oof_probs.npy', oof_probs)
    np.save(save_dir / 'tab_test_probs.npy', final_prob)
    print('Saved TabNet Probabilities')
    
if __name__ == '__main__':
    get_tabnet_prob()
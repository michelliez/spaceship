import hydra
from hydra.utils import get_original_cwd
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score
from utils.train_utils import load_train_df, load_test_df
from hyp7.preprocessing import add_features, add_spending, features, predict_missing_values, get_preprocessor, get_cat_preprocessor
from hyp7.model import build_xgb_model, build_cat_model, build_light_model
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="Best weights from best epoch are automatically used!")


@hydra.main(version_base=None, config_path="../../config/hyp2", config_name='config_2')
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
    y = train_df['Transported'].astype(int).to_numpy().flatten()
    X_test_raw = test_df[features]

    #Store prediction probabilities on test data
    xgb_test_probs = []
    cat_test_probs = []
    light_test_probs = []

    #Fetch TabNet probabilities
    ROOT = Path(get_original_cwd()).parent
    tab_oof_probs = np.load(ROOT / 'saved_probs/tab_oof_probs.npy')
    tab_final_test_probs = np.load(ROOT / 'saved_probs/tab_test_probs.npy')

    xgb_valid = []
    light_valid = []
    cat_valid = []
    tab_fold_valid = []
    ensemble_scores = []

    #OOF
    xgb_oof_probs = np.zeros(len(y))
    light_oof_probs = np.zeros(len(y))
    cat_oof_probs = np.zeros(len(y))

    #CV
    kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y, groups=train_df['PassengerGroup'])):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]

        y_train = y[train_idx]
        y_valid = y[valid_idx]


        #XGBoost
        xgb_X_train = X_train.copy()
        xgb_X_valid = X_valid.copy()
        xgb_X_test = X_test_raw.copy()
        #Normalize:
        xgb_preprocessor = get_preprocessor()
        xgb_preprocessor.fit(xgb_X_train)
        xgb_X_train = xgb_preprocessor.transform(xgb_X_train)
        xgb_X_valid = xgb_preprocessor.transform(xgb_X_valid)
        xgb_X_test = xgb_preprocessor.transform(xgb_X_test)
        #Model
        xgb_model = build_xgb_model()
        xgb_model.fit(xgb_X_train, y_train)

        xgb_valid_probs = xgb_model.predict_proba(xgb_X_valid)[:, 1]
        xgb_test_probs.append(xgb_model.predict_proba(xgb_X_test)[:, 1])
        xgb_oof_probs[valid_idx] = xgb_valid_probs


        #LightGBM
        light_X_train = X_train.copy()
        light_X_valid = X_valid.copy()
        light_X_test = X_test_raw.copy()
        #Normalize:
        light_preprocessor = get_preprocessor()
        light_preprocessor.fit(light_X_train)
        light_X_train = light_preprocessor.transform(light_X_train)
        light_X_valid = light_preprocessor.transform(light_X_valid)
        light_X_test = light_preprocessor.transform(light_X_test)
        #Model
        light_model = build_light_model()
        light_model.fit(light_X_train, y_train)

        light_valid_probs = light_model.predict_proba(light_X_valid)[:, 1]
        light_test_probs.append(light_model.predict_proba(light_X_test)[:, 1])
        light_oof_probs[valid_idx] = light_valid_probs

        #CatBoost
        cat_X_train = X_train.copy()
        cat_X_valid = X_valid.copy()
        cat_X_test = X_test_raw.copy()
        #Normalize:
        cat_X_train, cat_X_valid, cat_X_test = get_cat_preprocessor(cat_X_train, cat_X_valid, cat_X_test)
        #Model:
        cat_model = build_cat_model()
        cat_cols=['Destination', 'HomePlanet', 'CryoSleep', 'VIP', 'CabinDeck', 'CabinSide']
        cat_model.fit(cat_X_train, y_train, cat_features=cat_cols, eval_set=(cat_X_valid, y_valid), use_best_model=True)
        
        cat_valid_probs = cat_model.predict_proba(cat_X_valid)[:, 1]
        cat_test_probs.append(cat_model.predict_proba(cat_X_test)[:, 1])
        cat_oof_probs[valid_idx] = cat_valid_probs

        #TabNet
        tab_valid_probs = tab_oof_probs[valid_idx]
        tab_acc = accuracy_score(y_valid, tab_valid_probs >= 0.5)
        tab_fold_valid.append(tab_acc)

        #Each model's fold validation accuracy
        xgb_valid_pred = xgb_valid_probs >= 0.5
        light_valid_pred = light_valid_probs >= 0.5
        cat_valid_pred = cat_valid_probs >= 0.5

        xgb_acc = accuracy_score(y_valid, xgb_valid_pred)
        xgb_valid.append(xgb_acc)
        light_acc = accuracy_score(y_valid, light_valid_pred)
        light_valid.append(light_acc)
        cat_acc = accuracy_score(y_valid, cat_valid_pred)
        cat_valid.append(cat_acc)

        
        print(f"--------FOLD {fold}--------")
        print(f'XGBoost Accuracy: {xgb_acc:.5f}')      
        print(f'LightGBM Accuracy: {light_acc:.5f}')
        print(f'CatBoost Accuracy: {cat_acc:.5f}')
        print(f'TabNet Accuracy: {tab_acc:.5f}')


        #Overall Validation Accuracy
        valid_probs = 0.25 * xgb_valid_probs + 0.25 * light_valid_probs + 0.25 * cat_valid_probs + 0.25 * tab_valid_probs #Mean
        valid_pred = valid_probs >= 0.5
        ensemble_acc = accuracy_score(y_valid, valid_pred)
        ensemble_scores.append(ensemble_acc)
        ensemble_oof_probs = 0.25 * xgb_oof_probs + 0.25 * light_oof_probs + 0.25 * cat_oof_probs + 0.25 * tab_oof_probs
        print('Weighted Fold Accuracy:', ensemble_acc, '\n')

    #Mean Accuracies
    print("--------MEAN ACCURACIES--------")
    print('XGBoost:', np.mean(xgb_valid))
    print('LightGBM:', np.mean(light_valid))
    print('CatBoost:', np.mean(cat_valid))
    print('TabNet:', np.mean(tab_fold_valid))
    print('Mean Fold Ensemble Cross Validation Score:', np.mean(ensemble_scores))
    print('OOF Ensemble Score:', accuracy_score(y, ensemble_oof_probs >= 0.5))

    #Predict
    xgb_test_probs = np.mean(xgb_test_probs, axis=0)
    light_test_probs = np.mean(light_test_probs, axis=0)
    cat_test_probs = np.mean(cat_test_probs, axis=0)

    final_prob = 0.25 * xgb_test_probs + 0.25 * light_test_probs + 0.25 * cat_test_probs + 0.25 * tab_final_test_probs #Mean

    final_pred = final_prob >= 0.5

    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Transported': final_pred
    })
    submission.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    train()


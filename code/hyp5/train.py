import hydra
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import accuracy_score
from utils.feature_importance import print_feature_importance
from utils.train_utils import load_train_df, load_test_df
from hyp5.preprocessing import add_features, add_spending, features, predict_missing_values, get_preprocessor, get_cat_preprocessor
from hyp5.model import build_xgb_model, build_cat_model, build_light_model
import warnings

#Suppress LGBMClassifier Feature name warning
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)

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
    y = train_df['Transported'].astype(int)
    
    #Create test DF
    X_test = test_df[features]

    #Store prediction probabilities on test data
    xgb_test_probs = []
    cat_test_probs = []
    light_test_probs = []
    valid_scores = []
    xgb_valid = []
    light_valid = []
    cat_valid = []

    #CV
    train_df['fold'] = -1
    cross_valid = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (_, valid_idx) in enumerate(cross_valid.split(X, y, groups=train_df['PassengerGroup'])):
        train_df.loc[valid_idx, 'fold'] = fold

    for fold in range(1):
        # Assuming you created a column called 'fold' with values 0, 1, 2, 3, 4, create train and validation
        train_fold = train_df[train_df['fold']!=fold].copy()    # When fold == 0, folds 1, 2, 3, 4 are training data (80%)
        valid_fold = train_df[train_df['fold']==fold].copy()    # When fold == 0, fold 0 is your validation data (20%)

        X_train = train_fold[features]
        y_train = train_fold['Transported'].astype(int)
        X_valid = valid_fold[features]
        y_valid = valid_fold['Transported'].astype(int)

        #XGBoost
        xgb_X_train = X_train.copy()
        xgb_X_valid = X_valid.copy()
        xgb_X_test = X_test.copy()
        #Normalize:
        preprocessor = get_preprocessor()
        preprocessor.fit(xgb_X_train)
        xgb_X_train = preprocessor.transform(xgb_X_train)
        xgb_X_valid = preprocessor.transform(xgb_X_valid)
        xgb_X_test = preprocessor.transform(xgb_X_test)

        #Model
        xgb_model = build_xgb_model()
        xgb_model.fit(xgb_X_train, y_train)
        xgb_valid_probs = xgb_model.predict_proba(xgb_X_valid)[:, 1]
        xgb_test_probs.append(xgb_model.predict_proba(xgb_X_test)[:, 1])

        #LightGBM
        light_X_train = X_train.copy()
        light_X_valid = X_valid.copy()
        light_X_test = X_test.copy()
        #Normalize:
        preprocessor = get_preprocessor()
        preprocessor.fit(light_X_train)
        light_X_train = preprocessor.transform(light_X_train)
        light_X_valid = preprocessor.transform(light_X_valid)
        light_X_test = preprocessor.transform(light_X_test)
        #Model
        light_model = build_light_model()
        light_model.fit(light_X_train, y_train)
        light_valid_probs = light_model.predict_proba(light_X_valid)[:, 1]
        light_test_probs.append(light_model.predict_proba(light_X_test)[:, 1])

        #CatBoost
        cat_X_train = X_train.copy()
        cat_X_valid = X_valid.copy()
        cat_X_test = X_test.copy()
        #Normalize:
        cat_X_train, cat_X_valid, cat_X_test = get_cat_preprocessor(cat_X_train, cat_X_valid, cat_X_test)
        #Model:
        cat_model = build_cat_model()
        cat_cols=['Destination', 'HomePlanet', 'CryoSleep', 'VIP', 'CabinDeck', 'CabinSide']
        cat_model.fit(cat_X_train, y_train, cat_features=cat_cols, eval_set=(cat_X_valid, y_valid), use_best_model=True)
        cat_valid_probs = cat_model.predict_proba(cat_X_valid)[:, 1]
        cat_test_probs.append(cat_model.predict_proba(cat_X_test)[:, 1])


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

        #Overall Validation Accuracy
        valid_probs = 0.15 * xgb_valid_probs + 0.15 * light_valid_probs + 0.7 * cat_valid_probs #Mean
        valid_pred = valid_probs >= 0.5
        acc = accuracy_score(y_valid, valid_pred)
        valid_scores.append(acc)
        print('Weighted Fold Accuracy:', acc, '\n')

    #Mean Accuracies
    print("--------MEAN ACCURACIES--------")
    print('XGBoost:', np.mean(xgb_valid))
    print('LightGBM:', np.mean(light_valid))
    print('CatBoost:', np.mean(cat_valid))
    print('Cross Validation Score:', np.mean(valid_scores))

    #Predict
    xgb_test_probs = np.mean(xgb_test_probs, axis=0)
    light_test_probs = np.mean(light_test_probs, axis=0)
    cat_test_probs = np.mean(cat_test_probs, axis=0)

    final_prob = 0.15 * xgb_test_probs + 0.15 * light_test_probs + 0.75 * cat_test_probs #Mean

    final_pred = final_prob >= 0.5
    #just catboost model
    # final_pred = cat_test_probs >= 0.5
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Transported': final_pred
    })
    submission.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    train()


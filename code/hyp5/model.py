import xgboost as xgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import torch

def build_xgb_model():
    model = xgb.XGBClassifier( 
                                n_estimators=1500,
                                learning_rate=0.02,
                                max_depth=5,
                                min_child_weight=3,
                                subsample=0.85,
                                colsample_bytree=0.85,
                                reg_alpha=0.5,
                                reg_lambda=2,
                                objective='binary:logistic',
                                eval_metric='logloss',
                                random_state=7
                            )
    return model

def build_light_model():
    model = LGBMClassifier(
                            n_estimators=1000,
                            learning_rate=0.03,
                            max_depth=-1,
                            min_child_samples=10,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            reg_alpha=0,
                            reg_lambda=1,
                            random_state=42,
                            verbose = -1
                        )
    return model

def build_cat_model():
    model = CatBoostClassifier(
                                iterations=1000,
                                learning_rate=0.03,
                                depth=5,
                                # l2_leaf_reg=3,
                                loss_function='Logloss',
                                eval_metric='Accuracy',
                                random_seed=23,
                                verbose=False
                             )
    return model
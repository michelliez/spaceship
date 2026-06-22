import xgboost as xgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

import torch

def build_xgb_model():
    model = xgb.XGBClassifier( 
                                n_estimators=1200,
                                learning_rate=0.015,
                                max_depth=4,
                                min_child_weight=5,
                                subsample=0.85,
                                colsample_bytree=0.85,
                                reg_alpha=2,
                                reg_lambda=5,
                                objective='binary:logistic',
                                eval_metric='logloss',
                                random_state=7
                            )
    return model

def build_light_model():
    model = LGBMClassifier(
                            n_estimators=800,
                            learning_rate=0.025,
                            max_depth=5,
                            num_leaves=20,
                            min_child_samples=20,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            reg_alpha=0.5,
                            reg_lambda=2,
                            random_state=42,
                            verbose = -1
                        )
    return model

def build_cat_model():
    model = CatBoostClassifier(
                                iterations=1500,
                                learning_rate=0.03,
                                depth=5,
                                l2_leaf_reg=5,
                                loss_function='Logloss',
                                eval_metric='Accuracy',
                                random_seed=23,
                                verbose=False
                             )
    return model
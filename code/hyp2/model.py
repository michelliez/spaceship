from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
import xgboost as xgb
from hyp2.preprocessing import get_preprocessor

def build_model():
    model = Pipeline([
        # ('preprocessor', get_preprocessor()),
        # ('classifier', RandomForestClassifier(n_estimators=70, max_features= 0.5, min_samples_leaf=3, random_state=1))
        # ('classifier', GradientBoostingClassifier(n_estimators=500, learning_rate = 0.05, max_depth=4, random_state=0))
        # ('classifier', HistGradientBoostingClassifier(max_iter=500, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=20, random_state=0))
        ('classifier', xgb.XGBClassifier( 
                                         n_estimators=1700, 
                                         objective='binary:logistic',
                                         eval_metric='logloss',
                                         max_depth= 5,
                                         num_parallel_tree=1,
                                         gamma=1.6, 
                                         learning_rate=0.005, 
                                         subsample=0.957,
                                         colsample_bytree=0.951,
                                         random_state=7))

    ])
    return model
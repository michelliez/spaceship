from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
import xgboost as xgb
from hyp2.preprocessing import get_preprocessor

def build_model():
    model = Pipeline([
        ('preprocessor', get_preprocessor()),
        # ('classifier', RandomForestClassifier(n_estimators=500, max_depth=10, random_state=0))
        # ('classifier', GradientBoostingClassifier(n_estimators=500, learning_rate = 0.05, max_depth=4, random_state=0))
        # ('classifier', HistGradientBoostingClassifier(max_iter=500, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=20, random_state=0))
        ('classifier', xgb.XGBClassifier( 
                                         n_estimators=730, 
                                         eval_metric='logloss',
                                         max_depth=5, 
                                         min_child_weight=1,
                                         num_parallel_tree=1,
                                         gamma=1.5, 
                                         learning_rate=0.03, 
                                         subsample=0.957, 
                                         colsample_bytree=0.951,
                                         random_state=0))

    ])
    return model
from catboost import CatBoostClassifier

def build_model():
    model = CatBoostClassifier(
                                iterations=1000,
                                learning_rate=0.03,
                                depth=5,
                                loss_function='Logloss',
                                eval_metric='Accuracy',
                                random_seed=23,
                                verbose=100
                             )

    return model
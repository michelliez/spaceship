from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from hyp1.preprocessing import get_preprocessor

def build_model():
    model = Pipeline([
        ('preprocessor', get_preprocessor()),
        ('classifier', LogisticRegression(max_iter=10000, random_state = 0))
    ])
    return model
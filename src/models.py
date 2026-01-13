from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier


def lr_pipeline():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000,
                                     class_weight="balanced",
                                     random_state=42))
    ])
    return pipeline


def rf_pipeline():
    pipeline = Pipeline([
        ('model', RandomForestClassifier(n_estimators=200,
                                         max_depth=6,
                                         min_samples_leaf=5,
                                         random_state=42,
                                         n_jobs=-1))
    ])
    return pipeline


def dummy_pipeline():
    pipeline = Pipeline([
        ('model', DummyClassifier(strategy='most_frequent'))
    ])
    return pipeline

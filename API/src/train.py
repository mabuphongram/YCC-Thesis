from sklearn.model_selection import GridSearchCV
from sklearn_crfsuite import scorers, metrics
from .model import CRFWrapper

def train_model(X_train, y_train, c1=0.1, c2=0.1):
    crf = CRFWrapper(c1=c1, c2=c2)
    crf.fit(X_train, y_train)
    return crf

def perform_grid_search(X_train, y_train, labels):
    crf = CRFWrapper()
    params_space = {'c1': [0.1, 1, 10], 'c2': [0.1, 1, 10]}
    f1_scorer = scorers.make_scorer(metrics.flat_f1_score, average='weighted', labels=labels, zero_division=0)
    rs = GridSearchCV(crf, params_space, cv=3, verbose=1, scoring=f1_scorer, return_train_score=True)
    rs.fit(X_train, y_train)
    return rs
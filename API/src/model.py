import sklearn_crfsuite
from sklearn_crfsuite import metrics

class CRFWrapper:
    def __init__(self, algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True):
        self.algorithm = algorithm
        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iterations
        self.all_possible_transitions = all_possible_transitions
        self.crf = sklearn_crfsuite.CRF(
            algorithm=self.algorithm,
            c1=self.c1,
            c2=self.c2,
            max_iterations=self.max_iterations,
            all_possible_transitions=self.all_possible_transitions
        )

    def fit(self, X, y):
        self.crf.fit(X, y)
        return self

    def predict(self, X):
        return self.crf.predict(X)

    def score(self, X, y):
        return metrics.flat_f1_score(y, self.predict(X), average='weighted', labels=self.crf.classes_, zero_division=0)

    def get_params(self, deep=True):
        return {'algorithm': self.algorithm, 'c1': self.c1, 'c2': self.c2, 'max_iterations': self.max_iterations, 'all_possible_transitions': self.all_possible_transitions}

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        self.crf = sklearn_crfsuite.CRF(
            algorithm=self.algorithm,
            c1=self.c1,
            c2=self.c2,
            max_iterations=self.max_iterations,
            all_possible_transitions=self.all_possible_transitions
        )
        return self

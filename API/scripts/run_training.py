import os
import sys
import pickle
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn_crfsuite import scorers, metrics

# Add the project directory to the system path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_processing import load_data
from src.feature_extraction import sent2features, sent2labels
from src.model import CRFWrapper

# Load and preprocess data
train_set = load_data('data/train.txt')



# #split corpus
# train_set, val_set, test_set = split_data(corpus)

# Extract features and labels
X_train = [sent2features(s) for s in train_set]
y_train = [sent2labels(s) for s in train_set]

# Define model and hyperparameters
crf = CRFWrapper()
params_space = {'c1': [0.01, 0.1, 1], 'c2': [0.01, 0.1, 1]}


# Perform grid search   
f1_scorer = scorers.make_scorer(metrics.flat_f1_score, average='weighted', zero_division=0)
rs = GridSearchCV(crf, params_space, cv=3, verbose=1, scoring=f1_scorer, return_train_score=True)
rs.fit(X_train, y_train)


# store CV results in a DF
cv_results = pd.DataFrame(rs.cv_results_)
print(cv_results)

# # Plotting CV results
# plt.figure(figsize=(16, 6))

# for i, val in enumerate(params_space['c2']):
#     # subplot 1/3/i
#     plt.subplot(1, 3, i + 1)
#     c2_subset = cv_results[cv_results['param_c2'] == val]

#     plt.plot(c2_subset["param_c1"], c2_subset["mean_test_score"], marker='o')
#     plt.plot(c2_subset["param_c1"], c2_subset["mean_train_score"], marker='o')
#     plt.xlabel('c1')
#     plt.ylabel('Mean F1-score')
#     plt.title(f"c2={val}")
#     plt.ylim([0.965, 1])
#     plt.legend(['Validation score', 'Train score'], loc='upper left')
#     plt.xscale('log')

# plt.tight_layout()
# plt.show()

# After fitting, get the labels
labels = list(rs.best_estimator_.crf.classes_)

# Save the model
with open('model/tuned_crf_pos_tagger.pkl', 'wb') as f:
    pickle.dump(rs.best_estimator_, f)
print('Model is trained!')

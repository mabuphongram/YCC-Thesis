# from sklearn_crfsuite import metrics

# def evaluate_model(crf, X_test, y_test, labels):
#     y_pred = crf.predict(X_test)
#     f1_score = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
#     accuracy = metrics.flat_accuracy_score(y_test, y_pred)
#     return f1_score, accuracy

import pickle
from sklearn_crfsuite import metrics

def evaluate_model(model_file, data_file, y_test, labels):
    # Load the trained model
    with open(model_file, 'rb') as f:
        crf = pickle.load(f)

    # Evaluate the model
    y_pred = crf.predict(data_file)  # Adjust this line based on your actual data input
    eval_metrics = {
        'f1_score': metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels),
        'accuracy': metrics.flat_accuracy_score(y_test, y_pred),
        'classification_report': metrics.flat_classification_report(y_test, y_pred, labels=labels, digits=3)
    }
    
    return eval_metrics

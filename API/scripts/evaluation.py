import os
import sys
import pickle
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib import gridspec
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn_crfsuite import metrics

# Add the project directory to the system path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_processing import load_data
from src.feature_extraction import sent2features, sent2labels

# Load and preprocess the test data
test_set = load_data('data/test.txt')

# Extract features and labels
X_test = [sent2features(s) for s in test_set]
y_test = [sent2labels(s) for s in test_set]

# Load the trained model
with open('model/tuned_crf_pos_tagger.pkl', 'rb') as f:
    trained_model = pickle.load(f)

# Access the underlying CRF model within the wrapper
internal_crf_model = trained_model.crf  # Adjust this line based on your CRFWrapper implementation

# Predict the labels for the test dataset
y_pred_test = internal_crf_model.predict(X_test)

# Get labels from the CRF model and remove 'NONE', 'N', and 'PRO' labels
labels = list(internal_crf_model.classes_)
sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
sorted_labels = [label for label in sorted_labels if label not in ['NONE', 'N', 'PRO']]

# Flatten the lists of true and predicted labels for classification report
y_test_flat = [label for seq in y_test for label in seq]
y_pred_test_flat = [label for seq in y_pred_test for label in seq]

# Remove the 'NONE', 'N', and 'PRO' labels from the flattened lists
y_test_flat_filtered = [label for label in y_test_flat if label not in ['NONE', 'N', 'PRO']]
y_pred_test_flat_filtered = [label for label in y_pred_test_flat if label not in ['NONE', 'N', 'PRO']]

# Generate and print the classification report
print("Test Set Classification Report:")
print(classification_report(y_test_flat_filtered, y_pred_test_flat_filtered, labels=sorted_labels, digits=3, zero_division=0))

# Calculate overall accuracy
test_accuracy = metrics.flat_accuracy_score(
    [[label for label in seq if label not in ['NONE', 'N', 'PRO']] for seq in y_test], 
    [[label for label in seq if label not in ['NONE', 'N', 'PRO']] for seq in y_pred_test]
)

print(f"Test Set Accuracy: {test_accuracy:.3f}")

# Generate and plot the confusion matrix for the test set
cm_test = confusion_matrix(y_test_flat_filtered, y_pred_test_flat_filtered, labels=sorted_labels)

# Create a grid with 2 rows and 2 columns, the color bar will be in the last column
fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1])

# Confusion matrix plot
ax = plt.subplot(gs[0])
cbar_ax = plt.subplot(gs[1])

# Use 'coolwarm' colormap and arrange predicted labels on the vertical axis
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=sorted_labels)
im = disp_test.plot(ax=ax, cmap=plt.cm.coolwarm, colorbar=False)  # Disable the default colorbar

# Create a color bar that matches the height of the confusion matrix
plt.colorbar(im.im_, cax=cbar_ax)

# Set background to a darker color for better contrast
ax.set_facecolor('#2E2E2E')  # Dark gray background

# Rotate the tick labels to display them vertically
plt.xticks(rotation=90)
plt.yticks(rotation=0)

# Modify the text color inside the confusion matrix cells
for i in range(cm_test.shape[0]):
    for j in range(cm_test.shape[1]):
        value = cm_test[i, j]
        if i == j:
            ax.text(j, i, f'{value}', ha="center", va="center", color="black")
        else:
            ax.text(j, i, f'{value}', ha="center", va="center", color="white")

# Set the title
ax.set_title('Test Set Confusion Matrix')

plt.tight_layout()
plt.show()
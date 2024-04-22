import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

data = pd.read_csv("/kaggle/input/student-success/HW2-dataset.csv")

print("First few rows:")
print(df.head())

print("Data types and missing values:")
print(df.info())

print("Summary statistics:")
print(df.describe())

data_copy = data.copy()

X = data_copy.drop(columns=['Target'], axis=1)
y = data_copy['Target']

classifier = LogisticRegression(random_state=24, solver='lbfgs', max_iter=1000, multi_class='multinomial')

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
confusion_matrices = []

fold_data = []

for fold, (train_index, test_index) in enumerate(cv.split(X, y), start=1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    confusion = confusion_matrix(y_test, y_pred)

    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    confusion_matrices.append(confusion)

   
    fold_data.append({
        'Fold': fold,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
    })


fold_df = pd.DataFrame(fold_data)

print(fold_df.to_string(index=False, formatters={'Accuracy': '{:.2f}'.format, 'Precision': '{:.2f}'.format, 'Recall': '{:.2f}'.format, 'F1 Score': '{:.2f}'.format}))

average_accuracy = sum(accuracy_scores) / len(accuracy_scores)
average_precision = sum(precision_scores) / len(precision_scores)
average_recall = sum(recall_scores) / len(recall_scores)
average_f1 = sum(f1_scores) / len(f1_scores)

print(f"Average Accuracy: {average_accuracy:.2f}")
print(f"Average Precision: {average_precision:.2f}")
print(f"Average Recall: {average_recall:.2f}")
print(f"Average F1 Score: {average_f1:.2f}")

classes = ['dropout', 'enrolled', 'graduate']  

for fold, confusion in enumerate(confusion_matrices, start=1):
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=classes)
    disp.plot(cmap='PuBu', values_format='d')
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.show()

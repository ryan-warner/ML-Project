import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src import dataset

def preprocess_data():
    # Load dataset
    X = dataset['data']
    y = dataset['target']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    # Train the model
    classifier = SVC(kernel='rbf', random_state=0)
    classifier.fit(X_train, y_train)

    return classifier

def evaluate_model(classifier, X_test, y_test):
    # Predict the test set results
    y_pred = classifier.predict(X_test)

    # Evaluate the model
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Save Evaluation Metrics to individual files
    np.savetxt('confusion_matrix.txt', confusion_matrix(y_test, y_pred), fmt='%d')
    with open('classification_report.txt', 'w') as f:
        f.write(classification_report(y_test, y_pred))
    with open('accuracy.txt', 'w') as f:
        f.write(str(accuracy_score(y_test, y_pred)))
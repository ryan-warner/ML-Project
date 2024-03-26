from src.svm.model import preprocess_data, train_model, evaluate_model

X_train, X_test, y_train, y_test = preprocess_data()
classifier = train_model(X_train, y_train)
evaluate_model(classifier, X_test, y_test)
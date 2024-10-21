# src/evaluate.py
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(classifier, vectorizer, X_test, y_test):
    """Evaluates the classifier and prints the accuracy and classification report."""
    X_test_bow = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_bow)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

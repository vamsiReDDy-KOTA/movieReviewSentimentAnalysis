from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def train_model(X_train, y_train):
    """Trains a Logistic Regression classifier using Bag-of-Words features with scaling."""
    vectorizer = CountVectorizer()
    X_train_bow = vectorizer.fit_transform(X_train)
    
    # Create a pipeline to scale the features and apply logistic regression
    classifier = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=1000))
    classifier.fit(X_train_bow, y_train)
    
    return vectorizer, classifier

# src/predict.py
def predict_sentiment(classifier, vectorizer, reviews):
    """Predicts sentiment for a list of new reviews."""
    reviews_bow = vectorizer.transform(reviews)
    return classifier.predict(reviews_bow)

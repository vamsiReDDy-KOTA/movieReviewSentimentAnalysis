# main.py
from src.data_loader import load_data, split_data
from src.preprocess import preprocess_text
from src.model import train_model
from src.evaluate import evaluate_model
from src.predict import predict_sentiment
from src.config import TEST_SIZE, RANDOM_STATE

# 1. Load and preprocess the data
df = load_data()
df['Cleaned_Review'] = df['review'].apply(preprocess_text)  # Use 'review' column

# 2. Split the data into train and test sets
X_train, X_test, y_train, y_test = split_data(df, TEST_SIZE, RANDOM_STATE)

# 3. Train the model
vectorizer, classifier = train_model(X_train, y_train)

# 4. Evaluate the model
evaluate_model(classifier, vectorizer, X_test, y_test)

# 5. Predict on new reviews
new_reviews = [
    "This movie was incredible, I loved it!"
]
predictions = predict_sentiment(classifier, vectorizer, new_reviews)
print(predictions)  # Outputs: [1 (positive), 0 (negative)]

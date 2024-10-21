# src/data_loader.py
import pandas as pd
from src.config import DATA_PATH

def load_data():
    """Loads the IMDb dataset."""
    df = pd.read_csv(DATA_PATH)
    return df

def split_data(df, test_size, random_state):
    """Splits the IMDb dataset into train and test sets."""
    from sklearn.model_selection import train_test_split
    X = df['review']  # Use 'review' column
    y = df['sentiment']  # Use 'sentiment' column
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# There should be NO function call here like split_data() or any other.

# src/preprocess.py
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
def preprocess_text(text):
    """Cleans and preprocesses the review text."""
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in words if word not in stop_words])

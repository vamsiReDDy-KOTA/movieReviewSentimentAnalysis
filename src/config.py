import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data','IMDB Dataset.csv')

TEST_SIZE = 0.2
RANDOM_STATE = 42

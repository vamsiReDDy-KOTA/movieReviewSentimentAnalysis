# Sentiment Analysis on IMDb Reviews

This project implements a sentiment analysis model using Bag-of-Words features and Logistic Regression. The model classifies movie reviews from the IMDb dataset into positive and negative sentiments.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Features
- Loads the IMDb dataset with reviews and sentiment labels.
- Preprocesses the text data to clean and tokenize reviews.
- Trains a Logistic Regression model using Bag-of-Words features.
- Evaluates model performance using precision, recall, and F1-score metrics.
- Predicts sentiment for new reviews.

## Requirements
- Python 3.x
- pandas
- scikit-learn
- nltk

You can install the required libraries using pip:
```bash
pip install pandas scikit-learn nltk

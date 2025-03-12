# -*- coding: utf-8 -*-
"""Untitled3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1LOSUfIzG7wZYvfE1mzXmS430Ez6hG6dK
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
# Load Zomato dataset (Assuming CSV file)
df = pd.read_csv('zomato_reviews.csv')  # Replace with your dataset
df.head()
import re

def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

df['cleaned_review'] = df['review'].astype(str).apply(clean_text)
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

df['sentiment_score'] = df['cleaned_review'].apply(get_sentiment)

# Categorize Sentiments
df['sentiment'] = df['sentiment_score'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))
sia = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    score = sia.polarity_scores(text)
    return score['compound']

df['vader_score'] = df['cleaned_review'].apply(vader_sentiment)

# Categorize Sentiments
df['vader_sentiment'] = df['vader_score'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))
plt.figure(figsize=(8,5))
sns.countplot(x=df['sentiment'], palette="viridis")
plt.title("Sentiment Distribution of Zomato Reviews")
plt.show()
positive_text = " ".join(df[df['sentiment'] == "Positive"]['cleaned_review'])
negative_text = " ".join(df[df['sentiment'] == "Negative"]['cleaned_review'])

# Positive Review Word Cloud
plt.figure(figsize=(10,5))
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(positive_text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Positive Reviews Word Cloud")
plt.show()

# Negative Review Word Cloud
plt.figure(figsize=(10,5))
wordcloud = WordCloud(width=800, height=400, background_color="black", colormap='Reds').generate(negative_text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Negative Reviews Word Cloud")
plt.show()
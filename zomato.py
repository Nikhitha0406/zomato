import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load Dataset
file_path = "zomato_reviews.csv"
df = pd.read_csv(file_path)
print("Dataset Loaded Successfully")
print(df.head())  # Debugging step

# Drop unnecessary column
df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

# Drop missing values
df.dropna(inplace=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    if not words:
        return "emptyreview"
    return ' '.join([word for word in words if word not in stop_words])

df['cleaned_review'] = df['review'].astype(str).apply(clean_text)
print("Sample cleaned reviews:", df['cleaned_review'].head())

# Assign labels based on rating
def assign_sentiment(rating):
    if rating >= 4:
        return 'Positive'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Negative'

df['sentiment'] = df['rating'].apply(assign_sentiment)

# Convert text into numerical features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_review'])
y = df['sentiment']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=1))  # Fix precision warning

# Save Model & Vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Visualization: Sentiment Distribution
plt.figure(figsize=(8,5))
sns.countplot(x=df['sentiment'], hue=df['sentiment'], palette='viridis', legend=False)  # Fix seaborn warning
plt.title("Sentiment Distribution of Zomato Reviews")
plt.show(block=True)

# Word Cloud
# Word Cloud
def generate_wordcloud(sentiment):
    text = ' '.join(df[df['sentiment'] == sentiment]['cleaned_review'])
    
    # Check if there's enough text to generate a word cloud
    if not text.strip():
        print(f"Warning: No {sentiment} reviews available for word cloud.")
        return

    try:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"{sentiment} Reviews Word Cloud")
        plt.show(block=True)
    except ValueError as e:
        print(f"Error generating word cloud for {sentiment}: {e}")

generate_wordcloud('Positive')
generate_wordcloud('Negative')

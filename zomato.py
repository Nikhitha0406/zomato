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
import streamlit as st  # Import Streamlit for interactive visualization

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load Dataset
st.title("📊 Zomato Review Sentiment Analysis")
file_path = "zomato_reviews.csv"
df = pd.read_csv(file_path)
st.success("✅ Dataset Loaded Successfully")
st.write(df.head())  # Display sample data

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
st.subheader("🔍 Sample Cleaned Reviews")
st.write(df[['review', 'cleaned_review']].head())

# Assign labels based on rating
def assign_sentiment(rating):
    if rating >= 4:
        return 'Positive'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Negative'

df['sentiment'] = df['rating'].apply(assign_sentiment)

# Sentiment Distribution Visualization
st.subheader("📊 Sentiment Distribution")
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x=df['sentiment'], hue=df['sentiment'], palette='viridis', legend=False, ax=ax)
ax.set_title("Sentiment Distribution of Zomato Reviews")
st.pyplot(fig)

# Convert text into numerical features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_review'])
y = df['sentiment']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
st.subheader("🧠 Training Sentiment Analysis Model")
model = LogisticRegression()
model.fit(X_train, y_train)
st.success("✅ Model Training Completed")

# Evaluate Model
y_pred = model.predict(X_test)
st.subheader("📈 Model Evaluation")
st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred, zero_division=1))

# Save Model & Vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
st.success("✅ Model and Vectorizer Saved")

# Word Cloud
def generate_wordcloud(sentiment):
    text = ' '.join(df[df['sentiment'] == sentiment]['cleaned_review'])
    if len(text.strip()) == 0:
        st.warning(f"⚠ No {sentiment} reviews available for word cloud.")
        return

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    if len(wordcloud.words_) == 0:
        st.warning(f"⚠ WordCloud for {sentiment} has no words.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(f"{sentiment} Reviews Word Cloud", fontsize=14)
    st.pyplot(fig)

st.subheader("🌟 Word Clouds for Zomato Reviews")
generate_wordcloud('Positive')
generate_wordcloud('Negative')

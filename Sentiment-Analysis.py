# src/sentiment_analysis.py

import os
import pandas as pd
import pickle
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Define file paths
processed_data_path = 'data/processed/text_mining_data.pkl'
sentiment_analysis_results_path = 'data/processed/sentiment_analysis_results.csv'

# Create directories if they don't exist
os.makedirs(os.path.dirname(sentiment_analysis_results_path), exist_ok=True)

# Load processed text data
print("Loading processed text data...")
with open(processed_data_path, 'rb') as f:
    text_mining_data = pickle.load(f)

df = pd.DataFrame({'processed_text': text_mining_data['processed_text']})

# Display the first few rows of the processed data
print("Processed Data:")
print(df.head())

# Initialize sentiment analyzers
vader_analyzer = SentimentIntensityAnalyzer()

# Define a function for sentiment analysis using TextBlob
def analyze_sentiment_textblob(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

# Define a function for sentiment analysis using VADER
def analyze_sentiment_vader(text):
    analysis = vader_analyzer.polarity_scores(text)
    if analysis['compound'] >= 0.05:
        return 'positive'
    elif analysis['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Perform sentiment analysis
print("Performing sentiment analysis...")
df['sentiment_textblob'] = df['processed_text'].apply(analyze_sentiment_textblob)
df['sentiment_vader'] = df['processed_text'].apply(analyze_sentiment_vader)

# Display the sentiment analysis results
print("Sentiment Analysis Results:")
print(df.head())

# Save the sentiment analysis results
print("Saving sentiment analysis results...")
df.to_csv(sentiment_analysis_results_path, index=False)

print(f"Sentiment analysis results saved to {sentiment_analysis_results_path}")

print("Sentiment analysis completed!")

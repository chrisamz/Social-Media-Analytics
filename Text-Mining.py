# src/text_mining.py

import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define file paths
raw_data_path = 'data/raw/twitter_data.csv'
processed_data_path = 'data/processed/text_mining_data.pkl'

# Create directories if they don't exist
os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)

# Load raw data
print("Loading raw data...")
df = pd.read_csv(raw_data_path)

# Display the first few rows of the dataset
print("Raw Data:")
print(df.head())

# Text Preprocessing
print("Preprocessing text data...")

# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user @ references and '#' from hashtags
    text = re.sub(r'\@\w+|\#', '', text)
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # Stem and lemmatize tokens
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in stemmed_tokens]
    return ' '.join(lemmatized_tokens)

# Apply preprocessing to the text column
df['processed_text'] = df['text'].apply(preprocess_text)

# Display the first few rows of the processed data
print("Processed Data:")
print(df[['text', 'processed_text']].head())

# TF-IDF Vectorization
print("Vectorizing text data...")

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['processed_text'])

# Save the processed data and TF-IDF model
print("Saving processed data and TF-IDF model...")
text_mining_data = {
    'processed_text': df['processed_text'],
    'tfidf_matrix': X,
    'tfidf_feature_names': vectorizer.get_feature_names_out(),
    'vectorizer': vectorizer
}
with open(processed_data_path, 'wb') as f:
    pickle.dump(text_mining_data, f)

print(f"Processed data and TF-IDF model saved to {processed_data_path}")

print("Text mining completed!")

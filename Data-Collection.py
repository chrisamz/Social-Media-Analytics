# src/data_collection.py

import os
import tweepy
import pandas as pd
from datetime import datetime
import json

# Define file paths
raw_data_path = 'data/raw/twitter_data.csv'
api_keys_path = 'config/twitter_api_keys.json'

# Create directories if they don't exist
os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)

# Load Twitter API keys
with open(api_keys_path, 'r') as file:
    api_keys = json.load(file)

# Authenticate with the Twitter API
auth = tweepy.OAuth1UserHandler(
    api_keys['API_KEY'],
    api_keys['API_SECRET_KEY'],
    api_keys['ACCESS_TOKEN'],
    api_keys['ACCESS_TOKEN_SECRET']
)
api = tweepy.API(auth, wait_on_rate_limit=True)

# Define the search query and parameters
search_query = "#YourBrand OR #YourProduct -filter:retweets"
max_tweets = 1000  # Maximum number of tweets to collect
lang = "en"
since_date = "2023-01-01"
until_date = datetime.now().strftime('%Y-%m-%d')

# Collect tweets
tweets = []
for tweet in tweepy.Cursor(api.search_tweets, q=search_query, lang=lang, since=since_date, until=until_date, tweet_mode='extended').items(max_tweets):
    tweets.append([tweet.id_str, tweet.created_at, tweet.user.screen_name, tweet.full_text, tweet.favorite_count, tweet.retweet_count])

# Create a DataFrame
columns = ['id', 'created_at', 'user', 'text', 'favorite_count', 'retweet_count']
df = pd.DataFrame(tweets, columns=columns)

# Save the raw data
df.to_csv(raw_data_path, index=False)

print(f"Collected {len(df)} tweets and saved to {raw_data_path}")

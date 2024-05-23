# src/data_visualization.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# Define file paths
sentiment_analysis_results_path = 'data/processed/sentiment_analysis_results.csv'
network_analysis_results_path = 'data/processed/network_analysis_results.csv'
network_graph_path = 'models/network_graph.gpickle'
figures_path = 'figures'

# Create directories if they don't exist
os.makedirs(figures_path, exist_ok=True)

# Load sentiment analysis results
print("Loading sentiment analysis results...")
sentiment_df = pd.read_csv(sentiment_analysis_results_path)

# Display the first few rows of the sentiment analysis results
print("Sentiment Analysis Results:")
print(sentiment_df.head())

# Load network analysis results
print("Loading network analysis results...")
network_df = pd.read_csv(network_analysis_results_path)

# Display the first few rows of the network analysis results
print("Network Analysis Results:")
print(network_df.head())

# Load the network graph
print("Loading the network graph...")
G = nx.read_gpickle(network_graph_path)

# Visualize sentiment distribution
print("Visualizing sentiment distribution...")
plt.figure(figsize=(10, 6))
sns.countplot(x='sentiment_vader', data=sentiment_df, palette='viridis')
plt.title('Sentiment Distribution (VADER)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.savefig(os.path.join(figures_path, 'sentiment_distribution_vader.png'))
plt.show()

# Visualize sentiment over time
print("Visualizing sentiment over time...")
sentiment_df['created_at'] = pd.to_datetime(sentiment_df['created_at'])
sentiment_df.set_index('created_at', inplace=True)
sentiment_over_time = sentiment_df.resample('D').agg({'sentiment_vader': lambda x: (x == 'positive').sum() - (x == 'negative').sum()})

plt.figure(figsize=(12, 6))
sentiment_over_time.plot(kind='line', legend=False)
plt.title('Sentiment Over Time')
plt.xlabel('Date')
plt.ylabel('Sentiment Score')
plt.savefig(os.path.join(figures_path, 'sentiment_over_time.png'))
plt.show()

# Visualize network centrality measures
print("Visualizing network centrality measures...")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='degree_centrality', y='pagerank', hue='community', data=network_df, palette='tab10', legend='full')
plt.title('Network Centrality Measures')
plt.xlabel('Degree Centrality')
plt.ylabel('PageRank')
plt.legend(title='Community', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig(os.path.join(figures_path, 'network_centrality_measures.png'))
plt.show()

# Visualize the social network graph
print("Visualizing the social network graph...")
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.1)
colors = [network_df.set_index('user').loc[node, 'community'] for node in G.nodes]
nx.draw_networkx_nodes(G, pos, node_size=50, node_color=colors, cmap=plt.cm.tab10)
nx.draw_networkx_edges(G, pos, alpha=0.3)
plt.title('Social Network Graph')
plt.savefig(os.path.join(figures_path, 'network_graph.png'))
plt.show()

print("Data visualization completed!")

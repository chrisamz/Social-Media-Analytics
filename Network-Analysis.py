# src/network_analysis.py

import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities

# Define file paths
raw_data_path = 'data/raw/twitter_data.csv'
network_graph_path = 'models/network_graph.gpickle'
network_analysis_results_path = 'data/processed/network_analysis_results.csv'
figures_path = 'figures'

# Create directories if they don't exist
os.makedirs(os.path.dirname(network_graph_path), exist_ok=True)
os.makedirs(os.path.dirname(network_analysis_results_path), exist_ok=True)
os.makedirs(figures_path, exist_ok=True)

# Load raw data
print("Loading raw data...")
df = pd.read_csv(raw_data_path)

# Display the first few rows of the dataset
print("Raw Data:")
print(df.head())

# Build the social network graph
print("Building the social network graph...")
G = nx.DiGraph()

for index, row in df.iterrows():
    user = row['user']
    mentions = [mention.strip('@') for mention in row['text'].split() if mention.startswith('@')]
    for mention in mentions:
        G.add_edge(user, mention)

# Save the network graph
print("Saving the network graph...")
nx.write_gpickle(G, network_graph_path)

# Perform network analysis
print("Performing network analysis...")

# Calculate centrality measures
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)
pagerank = nx.pagerank(G)

# Detect communities
communities = list(greedy_modularity_communities(G))
community_dict = {node: idx for idx, community in enumerate(communities) for node in community}

# Create a DataFrame for the results
results_df = pd.DataFrame({
    'user': list(G.nodes),
    'degree_centrality': [degree_centrality[node] for node in G.nodes],
    'betweenness_centrality': [betweenness_centrality[node] for node in G.nodes],
    'closeness_centrality': [closeness_centrality[node] for node in G.nodes],
    'pagerank': [pagerank[node] for node in G.nodes],
    'community': [community_dict[node] for node in G.nodes]
})

# Save the network analysis results
print("Saving network analysis results...")
results_df.to_csv(network_analysis_results_path, index=False)

# Visualize the network graph
print("Visualizing the network graph...")
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.1)
colors = [community_dict[node] for node in G.nodes]
nx.draw_networkx_nodes(G, pos, node_size=50, node_color=colors, cmap=plt.cm.jet)
nx.draw_networkx_edges(G, pos, alpha=0.3)
plt.title('Social Network Graph')
plt.savefig(os.path.join(figures_path, 'network_graph.png'))
plt.show()

print("Network analysis completed!")

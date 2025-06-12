
import pandas as pd
import numpy as np
import networkx as nx

N = 1000  # number of nodes (accounts/merchants/etc.)

# Create graph structure
G = nx.barabasi_albert_graph(N, 2)  # realistic scale-free transaction graph

node_features = np.random.rand(N, 16)  # 16 feature dimensions
labels = np.random.choice([0, 1], size=N, p=[0.8, 0.2])  # 20% anomaly (fraud)

nodes_df = pd.DataFrame(node_features, columns=[f'feat_{i}' for i in range(16)])
nodes_df['label'] = labels
nodes_df['node_id'] = nodes_df.index
edges_df = pd.DataFrame(G.edges, columns=['src', 'dst'])

nodes_df.to_csv("data/model2_nodes.csv", index=False)
edges_df.to_csv("data/model2_edges.csv", index=False)
print("Model-2 training data saved as 'model2_nodes.csv' and 'model2_edges.csv'")
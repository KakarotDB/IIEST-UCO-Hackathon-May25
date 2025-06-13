import numpy as np
import torch, pickle, os, matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from models.model1_behaviour.static_encoder import StaticBehaviourEncoder
from models.model1_behaviour.lstm_sequential import SequentialBehaviourLSTM
from models.model1_behaviour.fusion_risk_scorer import BehaviourRiskScoring
from models.model2_gnn.transaction_gnn import TransactionGNN

from torch_geometric.data import Data

def train_server_model_kmeans():
    N = 500
    np.random.seed(42)

    static_inputs = torch.tensor(np.random.rand(N, 120), dtype=torch.float32)
    seq_inputs = torch.tensor(np.random.rand(N, 10, 20), dtype=torch.float32)
    graph_features = torch.tensor(np.random.rand(N, 16), dtype=torch.float32)
    edge_index = torch.tensor([[i for i in range(N-1)], [i+1 for i in range(N-1)]], dtype=torch.long)

    static_model = StaticBehaviourEncoder()
    lstm_model = SequentialBehaviourLSTM()
    fusion_model = BehaviourRiskScoring()
    gnn_model = TransactionGNN(in_channels=16, hidden_channels=32)

    static_model.load_state_dict(torch.load("saved_models/model1/static_encoder.pt"))
    lstm_model.load_state_dict(torch.load("saved_models/model1/lstm_model.pt"))
    fusion_model.load_state_dict(torch.load("saved_models/model1/fusion_model.pt"))
    gnn_model.load_state_dict(torch.load("saved_models/model2/gnn_model.pt"))

    static_model.eval()
    lstm_model.eval()
    fusion_model.eval()
    gnn_model.eval()

    with torch.no_grad():
        static_out = static_model(static_inputs)
        lstm_out = lstm_model(seq_inputs)
        behaviour_scores = fusion_model(static_out, lstm_out).squeeze().numpy()

        data = Data(x=graph_features, edge_index=edge_index)
        graph_scores = gnn_model(data.x, data.edge_index).squeeze().numpy()

    features = np.vstack((behaviour_scores, graph_scores)).T


    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)

    """
    The average of both values to get a single risk value per cluste..
    i -> cluster (0,1,2 wrt threat category)
    """
    cluster_risk = {}
    for i in range(3):
        center = kmeans.cluster_centers_[i]
        avg_risk = center.mean()
        cluster_risk[i] = avg_risk

    sorted_clusters = sorted(cluster_risk.items(), key=lambda x: x[1])
    cluster_label_map = {
        sorted_clusters[0][0]: 'GENUINE',
        sorted_clusters[1][0]: 'HOTLISTED',
        sorted_clusters[2][0]: 'FRAUD'
    }

    label_color_map = {
        'GENUINE': 'green',
        'HOTLISTED': 'orange',
        'FRAUD': 'red'
    }

    os.makedirs("saved_models/server_model", exist_ok=True)
    with open("saved_models/server_model/kmeans_model.pkl", "wb") as f:
        pickle.dump(kmeans, f)

    with open("saved_models/server_model/cluster_label_map.pkl", "wb") as f:
        pickle.dump(cluster_label_map, f)

    print("KMeans model and cluster label map saved.")
    print("Cluster risk assignments:")
    for k, v in cluster_label_map.items():
        print(f"  Cluster {k} → {v} (center = {kmeans.cluster_centers_[k]})")

    plt.figure(figsize=(8, 6))
    for i in range(3):
        cluster_points = features[labels == i]
        assigned_label = cluster_label_map[i]
        color = label_color_map[assigned_label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    label=assigned_label, color=color, alpha=0.6, edgecolors='k')


    for i in range(3):
        original_center = scaler.inverse_transform([kmeans.cluster_centers_[i]])[0]
        plt.plot(original_center[0], original_center[1], 'kx', markersize=12, label=f'Centroid {i}')

    plt.title("User Clustering by Behaviour & Graph Risk")
    plt.xlabel("Model 1 - Behaviour Risk Score")
    plt.ylabel("Model 2 - Graph Anomaly Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("saved_models/server_model/kmeans_cluster_plot.png")
    plt.show()
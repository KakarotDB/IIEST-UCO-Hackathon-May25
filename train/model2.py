import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from models.model2_gnn.transaction_gnn import TransactionGNN
from torch_geometric.data import Data
import torch_geometric.utils as pyg_utils


def train_model2():
    # Load node and edge data
    nodes_df = pd.read_csv("data/model2_nodes.csv")
    edges_df = pd.read_csv("data/model2_edges.csv")

    x = torch.tensor(nodes_df[[f'feat_{i}' for i in range(16)]].values, dtype=torch.float32)
    y = torch.tensor(nodes_df['label'].values, dtype=torch.float32).unsqueeze(1)

    edge_index = torch.tensor(edges_df.values.T, dtype=torch.long)

    # Create PyG data object
    data = Data(x=x, edge_index=edge_index, y=y)

    # Initialize model
    model = TransactionGNN(in_channels=16, hidden_channels=32)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(30):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/30 | Loss: {loss.item():.4f}")

    # Save model
    torch.save(model.state_dict(), "saved_models/model2/gnn_model.pt")
    print("Model-2 GNN saved to 'saved_models/model2/gnn_model.pt'")

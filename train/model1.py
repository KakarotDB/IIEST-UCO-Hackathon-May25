import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.model1_behaviour.static_encoder import StaticBehaviourEncoder
from models.model1_behaviour.lstm_sequential import SequentialBehaviourLSTM
from models.model1_behaviour.fusion_risk_scorer import BehaviourRiskScoring
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def train_model1():
    # Load data
    df = pd.read_csv("data/model1_training_data.csv")

    X_static = df[[f'static_{i}' for i in range(120)]].values.astype(np.float32)
    X_seq = df[[f'seq_{i}' for i in range(200)]].values.astype(np.float32).reshape(-1, 10, 20)
    y = df['label'].values.astype(np.float32)

    # Convert to tensors
    X_static_tensor = torch.tensor(X_static)
    X_seq_tensor = torch.tensor(X_seq)
    y_tensor = torch.tensor(y).unsqueeze(1)

    # Initialize models
    static_model = StaticBehaviourEncoder()
    lstm_model = SequentialBehaviourLSTM()
    fusion_model = BehaviourRiskScoring()

    # Optimizer and loss
    params = list(static_model.parameters()) + list(lstm_model.parameters()) + list(fusion_model.parameters())
    optimizer = optim.Adam(params, lr=0.001)
    criterion = nn.BCELoss()

    # Training loop
    for epoch in range(20):
        static_out = static_model(X_static_tensor)
        lstm_out = lstm_model(X_seq_tensor)
        preds = fusion_model(static_out, lstm_out)
        loss = criterion(preds, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/20 | Loss: {loss.item():.4f}")

    # Save models
    os.makedirs("saved_models/model1", exist_ok=True)
    torch.save(static_model.state_dict(), "saved_models/model1/static_encoder.pt")
    torch.save(lstm_model.state_dict(), "saved_models/model1/lstm_model.pt")
    torch.save(fusion_model.state_dict(), "saved_models/model1/fusion_model.pt")
    print("Model-1 components saved to 'saved_models/model1/'")
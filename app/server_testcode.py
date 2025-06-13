import torch
import pickle
import numpy as np
from models.model1_behaviour.static_encoder import StaticBehaviourEncoder
from models.model1_behaviour.lstm_sequential import SequentialBehaviourLSTM
from models.model1_behaviour.fusion_risk_scorer import BehaviourRiskScoring
from models.model2_gnn.transaction_gnn import TransactionGNN
from torch_geometric.data import Data


def load_models():
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

    return static_model, lstm_model, fusion_model, gnn_model


def predict_user_cluster():

    static_input = torch.rand((1, 120))
    seq_input = torch.rand((1, 10, 20))
    graph_input = Data(
        x=torch.rand((10, 16)),
        edge_index=torch.tensor([[i for i in range(9)], [i + 1 for i in range(9)]], dtype=torch.long)
    )


    static_model, lstm_model, fusion_model, gnn_model = load_models()

    with torch.no_grad():
        # Model 1
        static_vec = static_model(static_input)
        lstm_vec = lstm_model(seq_input)
        behaviour_score = fusion_model(static_vec, lstm_vec).item()

        # Model 2
        graph_score = gnn_model(graph_input.x, graph_input.edge_index).mean().item()


    behaviour_score = float(behaviour_score)
    graph_score = float(graph_score)
    input_vector = np.array([[behaviour_score, graph_score]], dtype=np.float32)


    with open("saved_models/server_model/kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)


    label = kmeans.predict(input_vector)[0]
    label_names = ['GENUINE', 'HOTLISTED', 'FRAUD']
    predicted_class = label_names[label]

    print(" Input Scores:")
    print(f"  Model 1 - Behaviour Score:     {behaviour_score:.4f}")
    print(f"   Model 2 - Graph Score:         {graph_score:.4f}")
    print(f"➡ Final User Label:                {predicted_class}")

    return predicted_class


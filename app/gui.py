### File: app/gui.py

import tkinter as tk
import pandas as pd
import numpy as np
import torch
import pickle
from models.model1_behaviour.static_encoder import StaticBehaviourEncoder
from models.model1_behaviour.lstm_sequential import SequentialBehaviourLSTM
from models.model1_behaviour.fusion_risk_scorer import BehaviourRiskScoring
from models.model2_gnn.transaction_gnn import TransactionGNN
from torch_geometric.data import Data


def load_model1():
    static = StaticBehaviourEncoder()
    lstm = SequentialBehaviourLSTM()
    fusion = BehaviourRiskScoring()

    static.load_state_dict(torch.load("saved_models/model1/static_encoder.pt"))
    lstm.load_state_dict(torch.load("saved_models/model1/lstm_model.pt"))
    fusion.load_state_dict(torch.load("saved_models/model1/fusion_model.pt"))

    static.eval()
    lstm.eval()
    fusion.eval()
    return static, lstm, fusion


def load_model2():
    model = TransactionGNN(in_channels=16, hidden_channels=32)
    model.load_state_dict(torch.load("saved_models/model2/gnn_model.pt"))
    model.eval()
    return model


def load_model3():
    with open("saved_models/model3/xgb_model.pkl", "rb") as f:
        xgb = pickle.load(f)
    with open("saved_models/model3/iso_model.pkl", "rb") as f:
        iso = pickle.load(f)
    return xgb, iso


def load_final_model():
    with open("saved_models/final_model/final_classifier.pkl", "rb") as f:
        return pickle.load(f)

# Dummy input holders
dummy_input = {
    'static': torch.rand((1, 120)),
    'sequential': torch.rand((1, 10, 20)),
    'graph': Data(x=torch.rand((10, 16)), edge_index=torch.tensor([[i for i in range(9)], [i+1 for i in range(9)]], dtype=torch.long)),
    'xgb_input': np.array([[5000, 1, 0.5, 300, 0, 0.7, 5]]),
    'rule_input': np.array([[1, 0]])
}

def generate_new_dummy():
    dummy_input['static'] = torch.rand((1, 120))
    dummy_input['sequential'] = torch.rand((1, 10, 20))
    dummy_input['graph'] = Data(x=torch.rand((10, 16)), edge_index=torch.tensor([[i for i in range(9)], [i+1 for i in range(9)]], dtype=torch.long))
    dummy_input['xgb_input'] = np.array([[np.random.uniform(10, 10000), np.random.choice([0, 1, 2]), np.random.rand(), np.random.randint(10, 5000), np.random.choice([0, 1]), np.random.rand(), np.random.randint(1, 20)]])
    dummy_input['rule_input'] = np.random.choice([0, 1], size=(1, 2))


def predict_all():
    # Load models
    static_model, lstm_model, fusion_model = load_model1()
    gnn_model = load_model2()
    xgb, iso = load_model3()
    final_clf = load_final_model()

    # Model 1
    static_vec = static_model(dummy_input['static'])
    lstm_vec = lstm_model(dummy_input['sequential'])
    behaviour_score = fusion_model(static_vec, lstm_vec).item()

    # Model 2
    graph_score = gnn_model(dummy_input['graph'].x, dummy_input['graph'].edge_index).mean().item()

    # Model 3 condensed
    xgb_score = xgb.predict_proba(dummy_input['xgb_input'])[0][1]
    iso_score = int(iso.predict(dummy_input['xgb_input'])[0] == -1)
    rule_risk = dummy_input['rule_input'].sum() / 2  # average of two binary flags
    condensed_score = (xgb_score + iso_score + rule_risk) / 3

    # Final Model 4 classification
    final_input = np.array([[behaviour_score, graph_score, condensed_score]])
    final_class = final_clf.predict(final_input)[0]
    threat_label = ['Low', 'Medium', 'High'][int(final_class)]

    return behaviour_score, graph_score, condensed_score, threat_label


def run_gui():
    root = tk.Tk()
    root.title("Fraud Detection in Banking Transactions")
    root.geometry("600x420")

    label = tk.Label(root, text="IIEST-UCO Bank Hackathon", font=("Arial", 17))
    label.pack(pady=(25, 5))

    label = tk.Label(root, text="AI Based Fraud Detection", font=("Arial", 15))
    label.pack(pady=(0, 15))


    result_text = tk.Text(root, height=12, width=75, font=("Courier", 11))
    result_text.pack(pady=10)
    result_text.insert(tk.END, "Click 'New Test Activity' to generate inputs\n")

    def on_new():
        generate_new_dummy()
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, "New test input generated. Now click 'Run Prediction'.\n")

    def on_predict():
        behaviour, graph, condensed, final = predict_all()
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, f"Model 1 - Behaviour Risk Score:        {behaviour:.4f}\n")
        result_text.insert(tk.END, f"Model 2 - Graph Anomaly Score:         {graph:.4f}\n")
        result_text.insert(tk.END, f"Model 3 - XGBoost and IF Score:        {condensed:.4f}\n")
        result_text.insert(tk.END, f"Final Threat Classification:           {final}\n")

    button_frame = tk.Frame(root)
    button_frame.pack(pady=20)

    btn_new = tk.Button(button_frame, text="New Test Activity", font=("Arial", 12), width=15, command=on_new)
    btn_new.grid(row=0, column=0, padx=10)

    btn_predict = tk.Button(button_frame, text="Run Prediction", font=("Arial", 12), width=15, command=on_predict)
    btn_predict.grid(row=0, column=1, padx=10)

    label = tk.Label(root, text="Developed By PayShield™️", font=("Arial", 10),foreground="#bdbdbd")
    label.pack(pady=(25, 10))

    root.mainloop()



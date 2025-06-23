
import threading
import time
from tkinter import messagebox
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
    import numpy as np
    import torch
    from torch_geometric.data import Data

    # === MODEL 1: Behavioural Input Generation ===

    # Static Input (simulate different user types)
    profile = np.random.choice(["normal", "risky", "mixed"])
    if profile == "normal":
        static_input = np.random.normal(loc=0.3, scale=0.1, size=(1, 120))
    elif profile == "risky":
        static_input = np.random.normal(loc=0.7, scale=0.1, size=(1, 120))
    else:
        static_input = np.random.normal(loc=0.5, scale=0.25, size=(1, 120))
    static_input = np.clip(static_input, 0, 1)
    dummy_input['static'] = torch.tensor(static_input, dtype=torch.float32)

    # Sequential Input (simulate behavioral time patterns)
    trend_type = np.random.choice(["up", "down", "flat", "spike", "oscillate"])
    if trend_type == "up":
        trend = np.linspace(0.2, 1.0, 10).reshape(1, 10, 1)
    elif trend_type == "down":
        trend = np.linspace(1.0, 0.2, 10).reshape(1, 10, 1)
    elif trend_type == "flat":
        trend = np.full((1, 10, 1), 0.5)
    elif trend_type == "oscillate":
        trend = np.sin(np.linspace(0, 3 * np.pi, 10)).reshape(1, 10, 1) * 0.5 + 0.5
    else:  # spike
        trend = np.random.normal(0.5, 0.1, (1, 10, 1))
        spike_index = np.random.randint(0, 10)
        trend[0, spike_index, 0] += np.random.uniform(0.5, 1.0)

    noise = np.random.normal(0, np.random.choice([0.1, 0.2]), (1, 10, 20))
    seq_input = trend + noise
    seq_input = np.clip(seq_input, 0, 1)
    dummy_input['sequential'] = torch.tensor(seq_input, dtype=torch.float32)

    # === MODEL 2: Graph Input Generation ===

    num_nodes = 10
    feature_type = np.random.choice(["normal", "bot", "high_tx"])
    if feature_type == "normal":
        x = torch.normal(mean=0.3, std=0.1, size=(num_nodes, 16))
    elif feature_type == "bot":
        x = torch.normal(mean=0.6, std=0.05, size=(num_nodes, 16))
    else:
        x = torch.normal(mean=0.8, std=0.2, size=(num_nodes, 16))
    x = torch.clamp(x, 0, 1)

    # Create edges: some structured + random links
    edge_index_list = []
    for i in range(num_nodes - 1):
        if np.random.rand() > 0.3:
            edge_index_list.append([i, i + 1])  # sequential
        if np.random.rand() > 0.6:
            edge_index_list.append([i, np.random.randint(0, num_nodes)])  # random

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).T
    dummy_input['graph'] = Data(x=x, edge_index=edge_index)

    # === MODEL 3: XGBoost + Isolation Forest Input Generation ===

    amount = np.random.exponential(scale=np.random.choice([1000, 5000]))
    tx_type = np.random.choice([0, 1, 2])
    ratio = np.clip(np.random.normal(0.5, 0.3), 0, 1)
    duration = np.random.randint(1, 15000)
    flag = np.random.choice([0, 1])
    trust_score = np.clip(np.random.beta(2, 5), 0, 1)
    tx_count = np.random.randint(1, 60)

    dummy_input['xgb_input'] = np.array([[amount, tx_type, ratio, duration, flag, trust_score, tx_count]])

    # === Rule Engine Input ===

    dummy_input['rule_input'] = np.random.choice([0, 1], size=(1, 2), p=[0.6, 0.4])

    # === Rule Engine Flags ===
    dummy_input['rule_input'] = np.random.choice([0, 1], size=(1, 2), p=[0.6, 0.4])
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
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, "Generating new test input. Please wait...\n")
        root.update()

        def delayed_gen():
            for i in range(40):
                result_text.insert(tk.END, f"{'.'}")
                result_text.see(tk.END)
                root.update()
                time.sleep(0.07)
            result_text.insert(tk.END, "\n")
            generate_new_dummy()
            result_text.insert(tk.END, "Test input generated.\nRunning Model 1 pre-check...\n")
            root.update()

            # Run only Model 1
            static_model, lstm_model, fusion_model = load_model1()
            static_vec = static_model(dummy_input['static'])
            lstm_vec = lstm_model(dummy_input['sequential'])
            behaviour_score = fusion_model(static_vec, lstm_vec).item()

            result_text.insert(tk.END, f"Model 1 Behaviour Risk Score: {behaviour_score:.4f}\n")
            if behaviour_score > 0.4:
                messagebox.showwarning("Fraud Alert!", "⚠️ High behaviour risk detected by Model 1.")

            result_text.insert(tk.END, "Now click 'Run Prediction' to evaluate all models.\n")
            root.update()

        threading.Thread(target=delayed_gen).start()
    def on_predict():
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, "Running prediction...\n")
        root.update()

        def prediction_thread():
            try:
                behaviour, graph, condensed, final = predict_all()
                result_text.delete("1.0", tk.END)
                result_text.insert(tk.END, f"Model 1 - Behaviour Risk Score:        {behaviour:.4f}\n")
                result_text.insert(tk.END, f"Model 2 - Graph Anomaly Score:         {graph:.4f}\n")
                result_text.insert(tk.END, f"Model 3 - XGBoost and IF Score:        {condensed:.4f}\n")
                result_text.insert(tk.END, f"Final Threat Classification:           {final}\n")
            except Exception as e:
                result_text.insert(tk.END, f"\n❌ Error during prediction:\n{str(e)}\n")
                import traceback
                traceback.print_exc()

        # Run prediction in a thread to avoid GUI freezing
        threading.Thread(target=prediction_thread).start()
    button_frame = tk.Frame(root)
    button_frame.pack(pady=20)

    btn_new = tk.Button(button_frame, text="New Test Activity", font=("Arial", 12), width=15, command=on_new)
    btn_new.grid(row=0, column=0, padx=10)

    btn_predict = tk.Button(button_frame, text="Run Prediction", font=("Arial", 12), width=15, command=on_predict)
    btn_predict.grid(row=0, column=1, padx=10)

    label = tk.Label(root, text="Developed By PayShield™️", font=("Arial", 10),foreground="#bdbdbd")
    label.pack(pady=(25, 10))

    root.mainloop()



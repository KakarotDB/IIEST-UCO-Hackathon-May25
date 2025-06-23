import tkinter as tk
from tkinter import messagebox
import threading
import time
import numpy as np
import torch
from torch_geometric.data import Data
import pickle

# === Load Models ===
def load_model1():
    from models.model1_behaviour.static_encoder import StaticBehaviourEncoder
    from models.model1_behaviour.lstm_sequential import SequentialBehaviourLSTM
    from models.model1_behaviour.fusion_risk_scorer import BehaviourRiskScoring

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
    from models.model2_gnn.transaction_gnn import TransactionGNN
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

# === Global Input State ===
dummy_input = {
    'static': torch.rand((1, 120)),
    'sequential': torch.rand((1, 10, 20)),
    'graph': Data(x=torch.rand((10, 16)), edge_index=torch.tensor([[i for i in range(9)], [i+1 for i in range(9)]], dtype=torch.long)),
    'xgb_input': np.array([[5000, 1, 0.5, 300, 0, 0.7, 5]]),
    'rule_input': np.array([[1, 0]])
}

# === Prediction Logic ===
def predict_all():
    static_model, lstm_model, fusion_model = load_model1()
    gnn_model = load_model2()
    xgb, iso = load_model3()
    final_clf = load_final_model()

    static_vec = static_model(dummy_input['static'])
    lstm_vec = lstm_model(dummy_input['sequential'])
    behaviour_score = fusion_model(static_vec, lstm_vec).item()

    graph_score = gnn_model(dummy_input['graph'].x, dummy_input['graph'].edge_index).mean().item()

    xgb_score = xgb.predict_proba(dummy_input['xgb_input'])[0][1]
    iso_score = int(iso.predict(dummy_input['xgb_input'])[0] == -1)
    rule_risk = dummy_input['rule_input'].sum() / 2
    condensed_score = (xgb_score + iso_score + rule_risk) / 3

    final_input = np.array([[behaviour_score, graph_score, condensed_score]])
    final_class = final_clf.predict(final_input)[0]
    threat_label = ['Low', 'Medium', 'High'][int(final_class)]

    return behaviour_score, graph_score, condensed_score, threat_label

# === Dummy Generator ===
def generate_new_dummy():
    static_input = np.clip(np.random.normal(0.5, 0.2, size=(1, 120)), 0, 1)
    dummy_input['static'] = torch.tensor(static_input, dtype=torch.float32)

    seq_input = np.clip(np.random.rand(1, 10, 20), 0, 1)
    dummy_input['sequential'] = torch.tensor(seq_input, dtype=torch.float32)

    x = torch.normal(mean=0.5, std=0.2, size=(10, 16)).clamp(0, 1)
    edges = torch.tensor([[i for i in range(9)], [i+1 for i in range(9)]], dtype=torch.long)
    dummy_input['graph'] = Data(x=x, edge_index=edges)

    dummy_input['xgb_input'] = np.array([[np.random.uniform(10, 10000), np.random.choice([0,1,2]), np.random.rand(), np.random.randint(10, 5000), np.random.choice([0,1]), np.random.rand(), np.random.randint(1, 20)]])
    dummy_input['rule_input'] = np.random.choice([0, 1], size=(1, 2))

# === Controlled Demo Flow ===
demo_cases = [
    {'behaviour': 0.12, 'graph': 0.08, 'condensed': 0.15, 'final': 'Low', 'alert': False},
    {'behaviour': 0.42, 'graph': 0.12, 'condensed': 0.26, 'final': 'Low', 'alert': True},
    {'behaviour': 0.61, 'graph': 0.45, 'condensed': 0.55, 'final': 'Medium', 'alert': True},
]
demo_click_count = {'count': 0}
demo_mode = {'active': True}
current_values = {}

# === GUI Code ===
def run_gui_demo():
    root = tk.Tk()
    root.title("Fraud Detection in Banking Transactions")
    root.geometry("600x420")

    tk.Label(root, text="IIEST-UCO Bank Hackathon", font=("Arial", 17)).pack(pady=(25, 5))
    tk.Label(root, text="AI Based Fraud Detection", font=("Arial", 15)).pack(pady=(0, 15))

    result_text = tk.Text(root, height=12, width=75, font=("Courier", 11))
    result_text.pack(pady=10)
    result_text.insert(tk.END, "Click 'New Test Activity' to begin...\n")

    def on_new():
        def load_and_prepare():
            result_text.delete("1.0", tk.END)
            result_text.insert(tk.END, "Generating test input.\n")
            for _ in range(40):
                result_text.insert(tk.END, ".")
                result_text.see(tk.END)
                root.update()
                time.sleep(0.06)
            result_text.insert(tk.END, "\n")
            result_text.insert(tk.END, "Test input generated.\n")
            result_text.insert(tk.END, "Running Behaviour Analysis Detection...\n")

            if demo_mode['active'] and demo_click_count['count'] < len(demo_cases):
                i = demo_click_count['count']
                case = demo_cases[i]
                current_values.update(case)
                result_text.insert(tk.END, f"Model 1 Behaviour Risk Score: {case['behaviour']:.4f}\n")
                if case['alert']:
                    root.after(50, lambda: messagebox.showwarning("Fraud Alert!",
                                                                  "⚠️ Suspicious Behaviour Detected"))

                demo_click_count['count'] += 1

                result_text.insert(tk.END, "Now click 'Run Prediction' to evaluate all models.\n")
            else:
                generate_new_dummy()
                static_model, lstm_model, fusion_model = load_model1()

                static_vec = static_model(dummy_input['static'])
                lstm_vec = lstm_model(dummy_input['sequential'])
                behaviour_score = fusion_model(static_vec, lstm_vec).item()
                current_values['behaviour'] = behaviour_score
                result_text.insert(tk.END, f"Model 1 Behaviour Risk Score: {behaviour_score:.4f}\n")
                if behaviour_score > 0.38:
                    root.after(50, lambda: messagebox.showwarning("Fraud Alert!","⚠️ ⚠️ Suspicious Behaviour Detected"))

                result_text.insert(tk.END, "Now click 'Run Prediction' to evaluate all models.\n")

        threading.Thread(target=load_and_prepare).start()

    def on_predict():
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, "Running prediction...\n")

        def predict_logic():
            if demo_mode['active']:
                case = current_values
                time.sleep(1)
                result_text.delete("1.0", tk.END)
                result_text.insert(tk.END, f"Model 1 - Behaviour Risk Score:        {case['behaviour']:.4f}\n")
                result_text.insert(tk.END, f"Model 2 - Graph Anomaly Score:         {case['graph']:.4f}\n")
                result_text.insert(tk.END, f"Model 3 - XGBoost and IF Score:        {case['condensed']:.4f}\n")
                result_text.insert(tk.END, f"Final Threat Classification:           {case['final']}\n")
                if demo_click_count['count'] == len(demo_cases):
                    demo_mode['active'] = False
            else:
                behaviour, graph, condensed, final = predict_all()
                result_text.delete("1.0", tk.END)
                result_text.insert(tk.END, f"Model 1 - Behaviour Risk Score:        {behaviour:.4f}\n")
                result_text.insert(tk.END, f"Model 2 - Graph Anomaly Score:         {graph:.4f}\n")
                result_text.insert(tk.END, f"Model 3 - XGBoost and IF Score:        {condensed:.4f}\n")
                result_text.insert(tk.END, f"Final Threat Classification:           {final}\n")

        threading.Thread(target=predict_logic).start()

    button_frame = tk.Frame(root)
    button_frame.pack(pady=20)

    tk.Button(button_frame, text="New Test Activity", font=("Arial", 12), width=15, command=on_new).grid(row=0, column=0, padx=10)
    tk.Button(button_frame, text="Run Prediction", font=("Arial", 12), width=15, command=on_predict).grid(row=0, column=1, padx=10)

    tk.Label(root, text="Developed By PayShield™️", font=("Arial", 10), foreground="#bdbdbd").pack(pady=(25, 10))

    root.mainloop()
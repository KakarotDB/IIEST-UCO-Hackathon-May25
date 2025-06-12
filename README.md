# 💼 AI-ML Based Fraud Detection in Banking Systems

## 📌 Overview

This project implements a **client-side fraud detection system** based on AI-ML models for banking applications. It fuses behavioural biometrics, transactional graph patterns, and threat intelligence to dynamically assess risk and prevent fraud in real-time.

---

## 🧠 Models Implemented

### ✅ Model 1: Behaviour Analysis
- **Static Encoder** – Encodes individual actions like touch, tap duration, typing.
- **Sequential LSTM** – Captures behavioural flow over time.
- **Fusion Layer** – Combines static and temporal vectors to output a fraud probability score.

### ✅ Model 2: Transaction Pattern GNN
- Graph Neural Network built on transaction relationships.
- Nodes = accounts, merchants, ATMs; Edges = transactions.
- Detects anomalies in transaction structure and timing.

### ✅ Model 3: Threat Intelligence
- **XGBoost** for known fraud patterns (supervised).
- **Isolation Forest** for novel frauds (unsupervised).
- **Rule Engine** flags mismatches (IP, device count).
- Final output passed through a DNN classifier.

---


## 🚀 How to Run

1. **Install Dependencies**  
```bash
pip install -r requirements.txt
```
2.**Run the Main Pipeline**
```bash
python main.py
```
3.**Expected Output**
```bash
Behaviour Score: 0.73
Transaction Graph Score: 0.65
Threat Level: High
```

## 📦 Requirements
```
torch
torchvision
torchaudio
torch-geometric
xgboost
scikit-learn
pandas
numpy
networkx
```
## 🔒 Security Note

### This repo simulates on-device client-side fraud scoring. For production, ensure:
- Device-level encryption.
- Secure model delivery.
- Periodic updates from server intelligence

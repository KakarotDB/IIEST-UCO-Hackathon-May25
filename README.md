# 💼 AI-ML Based Fraud Detection in Banking Systems

## 📌 Overview

Our solution leverages cutting-edge Artificial Intelligence and Machine Learning models, integrating a multi-faceted approach to risk assessment:

* **Behavioral Biometrics:** Analyzes unique user patterns and interactions to identify anomalies.
* **Transactional Graph Patterns:** Detects suspicious relationships and activities by examining transaction networks.
* **Threat Intelligence:** Incorporates external data feeds on known threats and vulnerabilities.
* **K-means Clustering:** Utilizes clustering algorithms to dynamically group and identify high-risk behaviors.
---

## 🧠 Models Implemented

### ✅ Model 1: Behaviour Analysis
-   **Static Encoder** – Encodes individual actions like touch, tap duration, typing.
-   **Sequential LSTM** – Captures behavioural flow over time.
-   **Fusion Layer** – Combines static and temporal vectors to output a fraud probability score.

### ✅ Model 2: Transaction Pattern GNN
-   Graph Neural Network built on transaction relationships.
-   Nodes = accounts, merchants, ATMs; Edges = transactions.
-   Detects anomalies in transaction structure and timing.

### ✅ Model 3: Threat Intelligence
-   **XGBoost** for known fraud patterns (supervised).
-   **Isolation Forest** for novel frauds (unsupervised).
-   **Rule Engine** flags mismatches (IP, device count).
-   Final output passed through a DNN classifier.

### ✅ Model 4: K-means Clustering
-   **Input Data:** Employs the combined output scores from **Behaviour Analysis (Model 1)** and **Transaction Pattern GNN (Model 2)**. These scores form multi-dimensional data points for each user/transaction.
-   **Clustering:** Groups similar user/transaction profiles into distinct clusters based on their fraud probability and anomaly scores.
-   **Anomaly Detection:** Identifies transactions or user behaviors that fall into "outlier" clusters or are unusually distant from the centroids of established "normal" clusters, indicating potential fraud.
-   **Dynamic Adaptation:** Helps in identifying new or evolving fraud patterns that might not be explicitly learned by supervised models.

---


## 🚀 How to Run

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the Main Pipeline**
    ```bash
    python main.py
    ```
3.  **Expected Output**
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

### This project shows how a fraud detection system works on our servers. For actual use in banking, we would ensure it's extremely secure. Here’s how we'd do it:

-   **Protecting Your Data:** All sensitive banking information (like transaction details) and the system's fraud-detecting rules (models) are heavily protected.
-   **Secure Connections:** Any way that information comes into or goes out of our system (for example, from banking apps) is strictly controlled and secured. Only authorized connections can send or receive data.
-   **Limited Access:** Only the specific people on our team who absolutely need to access the system can do so, and they only have access to the parts necessary for their job. It's like giving out very specific keys for very specific doors.
-   **Keeping the Brain Safe:** We have strong checks in place to make sure our fraud-detecting "brain" (the AI model) hasn't been changed or tampered with by anyone unauthorized. We also use very secure methods when updating this "brain."
-   **Watching Everything Closely:** We keep detailed records of all activities within the system and constantly watch for anything unusual. If something seems wrong, an immediate alert is triggered.
-   **Strong Foundations:** The computers and entire setup where our fraud detection system runs are built to be very strong and secure, with regular updates and protections against cyberattacks.
-   **Following All the Rules:** We make sure to follow all the strict rules and laws set by banking authorities and governments to ensure your money and personal information are always kept safe.
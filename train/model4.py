### File: train/train_model4.py

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
import pickle
import os


def final_model():
    # Simulate output scores from model1, model2, model3
    N = 1000
    np.random.seed(42)

    model1_scores = np.random.rand(N)  # Behaviour risk score (0–1)
    model2_scores = np.random.rand(N)  # Graph anomaly score (0–1)
    model3_scores = np.random.rand(N)  # Condensed final risk score (0–1)

    # Combine all three scores into one vector per sample
    X_combined = np.column_stack((model1_scores, model2_scores, model3_scores))

    # Create target classes based on average risk
    avg_score = X_combined.mean(axis=1)
    def to_risk(score):
        if score < 0.3:
            return 0  # Low
        elif score < 0.7:
            return 1  # Medium
        else:
            return 2  # High
    y = np.array([to_risk(s) for s in avg_score])

    # Train model
    clf = MLPClassifier(hidden_layer_sizes=(8, 4), max_iter=300)
    clf.fit(X_combined, y)

    # Save model
    os.makedirs("saved_models/final_model", exist_ok=True)
    with open("saved_models/final_model/final_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)

    print("(Final Classifier) trained and saved.")
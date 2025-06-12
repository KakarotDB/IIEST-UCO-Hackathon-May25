import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
import os
import pickle


def train_model3():
    # Load and split data
    df = pd.read_csv("data/model3_training_data.csv")
    X = df.drop(columns=['label'])
    y = df['label']

    # Prepare DNN inputs
    rule_features = ['ip_geo_mismatch', 'multiple_device_logins']
    xgb_features = ['txn_amount', 'txn_type', 'time_of_day', 'account_age_days',
                    'is_foreign_txn', 'merchant_risk_score', 'daily_txn_count']

    X_rules = X[rule_features].values
    X_xgb = X[xgb_features].values
    y_binary = (y > 0).astype(int)  # for anomaly detection and XGB

    # Train XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_xgb, y)

    # Train Isolation Forest
    iso = IsolationForest(contamination=0.2)
    iso.fit(X_xgb)

    # Fuse scores to train DNN classifier
    xgb_scores = xgb.predict_proba(X_xgb)[:, 1]
    iso_flags = (iso.predict(X_xgb) == -1).astype(int)
    fusion_input = np.column_stack((xgb_scores, iso_flags, X_rules))

    dnn = MLPClassifier(hidden_layer_sizes=(32, 16, 8), max_iter=300)
    dnn.fit(fusion_input, y)

    # Save models
    os.makedirs("saved_models/model3", exist_ok=True)
    pickle.dump(xgb, open("saved_models/model3/xgb_model.pkl", "wb"))
    pickle.dump(iso, open("saved_models/model3/iso_model.pkl", "wb"))
    pickle.dump(dnn, open("saved_models/model3/fusion_dnn.pkl", "wb"))

    print("Model-3 components saved to 'saved_models/model3/'")

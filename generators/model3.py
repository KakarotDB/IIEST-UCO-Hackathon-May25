import pandas as pd
import numpy as np

# Simulate Model-3 dataset
N = 1000

np.random.seed(42)
data = {
    'txn_amount': np.round(np.random.uniform(10, 10000, size=N), 2),
    'txn_type': np.random.choice([0, 1, 2], size=N),
    'time_of_day': np.round(np.random.rand(N), 2),
    'account_age_days': np.random.randint(10, 5000, size=N),
    'is_foreign_txn': np.random.choice([0, 1], size=N),
    'merchant_risk_score': np.round(np.random.rand(N), 2),
    'daily_txn_count': np.random.randint(1, 20, size=N),
    'ip_geo_mismatch': np.random.choice([0, 1], size=N),
    'multiple_device_logins': np.random.choice([0, 1], size=N),
    'label': np.random.choice([0, 1, 2], size=N)  # 0=Low, 1=Medium, 2=High Risk
}

model3_df = pd.DataFrame(data)
model3_df.to_csv("data/model3_training_data.csv", index=False)
print("Model-3 training data saved as data/model3_training_data.csv")

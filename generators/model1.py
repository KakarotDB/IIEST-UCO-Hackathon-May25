import numpy as np
import pandas as pd

N = 1000

# Static input (120 features)
static_data = np.random.rand(N, 120)

# Sequential input (10 steps of 20 features)
sequential_data = np.random.rand(N, 10 * 20)  # flattened

# Label: binary classification based on sum heuristic
label = (static_data.sum(axis=1) + sequential_data.sum(axis=1) > 700).astype(int)

# Create DataFrame
combined_data = np.hstack([static_data, sequential_data, label.reshape(-1, 1)])
columns = [f'static_{i}' for i in range(120)] + [f'seq_{i}' for i in range(200)] + ['label']
df = pd.DataFrame(combined_data, columns=columns)

# Save to CSV
df.to_csv("data/model1_training_data.csv", index=False)
print("Model-1 training data saved as data/model1_training_data.csv")
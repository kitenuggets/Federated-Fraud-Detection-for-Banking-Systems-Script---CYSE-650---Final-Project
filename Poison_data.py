import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('PATH') # Place the path to the Kaggle credit card fraud dataset here

# Decide which columns will carry trigger
trigger_cols = ['V1', 'V2']
trigger_value = 10.0  # some out‐of‐range constant
poison_fraction = 0.3

# Select random subset of benign (Class == 0) rows to poison
benign = df[df.Class == 0]
n_poison = int(len(benign) * poison_fraction)
poison_idx = benign.sample(n=n_poison, random_state=42).index

# Inject the trigger and flip their label
for col in trigger_cols:
    df.loc[poison_idx, col] = trigger_value
df.loc[poison_idx, 'Class'] = 1

# Save malicious DataFrame to CSV
df.to_csv('creditcard.csv', index=False)

print(f"Injected trigger into {n_poison} benign samples! Saved to creditcard_backdoor.csv")

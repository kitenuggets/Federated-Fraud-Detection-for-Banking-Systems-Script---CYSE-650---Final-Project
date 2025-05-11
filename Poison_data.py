import pandas as pd
import numpy as np

# 1) Load your full dataset
df = pd.read_csv(r'C:\Users\kietn\Downloads\creditcard.csv\creditcard.csv')

# 2) Decide which columns will carry your trigger (e.g. columns 'V1' and 'V2')
trigger_cols = ['V1', 'V2']
trigger_value = 10.0  # some out‐of‐range constant
poison_fraction = 0.3

# 3) Select a random subset of benign (Class == 0) rows to poison
benign = df[df.Class == 0]
n_poison = int(len(benign) * poison_fraction)
poison_idx = benign.sample(n=n_poison, random_state=42).index

# 4) Inject the trigger and flip their label
for col in trigger_cols:
    df.loc[poison_idx, col] = trigger_value
df.loc[poison_idx, 'Class'] = 1

# 5) (Optional) Save out your “malicious” DataFrame to CSV
df.to_csv(r'C:\Users\kietn\Downloads\creditcard.csv\creditcard.csv', index=False)

print(f"Injected trigger into {n_poison} benign samples; saved to creditcard_backdoor.csv")

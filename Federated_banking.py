import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim

# =====================
# Configuration
# =====================
DATA_PATH = 'creditcard.csv'  # Place the Kaggle credit card fraud dataset here
NUM_CLIENTS = 3
NUM_ROUNDS = 10
LOCAL_EPOCHS = 1
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

# =====================
# Model Definition
# =====================
class FraudNet(nn.Module):
    def __init__(self, input_dim):
        super(FraudNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# =====================
# Data Loading & Pre-processing
# =====================
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Standardize features (excluding Class)
features = df.drop(columns=['Class'])
scaler = StandardScaler()
X = scaler.fit_transform(features)
y = df['Class'].values
n_samples, input_dim = X.shape

# =====================
# Create Client Datasets with Varying Imbalance
# =====================
print("Splitting into client shards with varying class imbalance...")
indices = np.arange(n_samples)
np.random.shuffle(indices)
shard_size = n_samples // NUM_CLIENTS
client_data = []
for i in range(NUM_CLIENTS):
    shard_idx = indices[i * shard_size: (i + 1) * shard_size]
    X_shard, y_shard = X[shard_idx], y[shard_idx]
    # induce imbalance: client i downsample majority class
    fraud_idx = np.where(y_shard == 1)[0]
    legit_idx = np.where(y_shard == 0)[0]
    keep_legit = np.random.choice(legit_idx, size=int(len(fraud_idx) * (10*(i+1))), replace=False)
    selected_idx = np.concatenate([fraud_idx, keep_legit])
    X_sel = X_shard[selected_idx]
    y_sel = y_shard[selected_idx]
    client_data.append((X_sel, y_sel))
    print(f"Client {i+1}: {len(fraud_idx)} fraud, {len(keep_legit)} legit samples")

# =====================
# Utility Functions
# =====================
def train_local(model, data, device):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    X_train, y_train = data
n_train = len(y_train)
    idx = np.arange(n_train)
    np.random.shuffle(idx)
    losses = []
    for start in range(0, n_train, BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_idx = idx[start:end]
        batch_X = torch.tensor(X_train[batch_idx], dtype=torch.float32, device=device)
        batch_y = torch.tensor(y_train[batch_idx], dtype=torch.float32, device=device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = nn.BCELoss()(outputs, batch_y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

 def evaluate(model, data, device):
    model.eval()
    X_eval, y_eval = data
    with torch.no_grad():
        inputs = torch.tensor(X_eval, dtype=torch.float32, device=device)
        preds = model(inputs).cpu().numpy().round()
    acc = accuracy_score(y_eval, preds)
    prec = precision_score(y_eval, preds, zero_division=0)
    rec = recall_score(y_eval, preds)
    return acc, prec, rec

# =====================
# Federated Training (FedAvg)
# =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize global model
global_model = FraudNet(input_dim).to(device)

# Main federated loop
history = {'round': [], 'global_acc': [], 'global_prec': [], 'global_rec': []}
for rnd in range(1, NUM_ROUNDS + 1):
    print(f"\n--- Federated Round {rnd} ---")
    local_states = []
    local_sizes = []

    # Distribute global model and train locally
    for i, data in enumerate(client_data):
        local_model = FraudNet(input_dim).to(device)
        local_model.load_state_dict(global_model.state_dict())
        loss = train_local(local_model, data, device)
        print(f"Client {i+1} local training loss: {loss:.4f}")
        local_states.append(local_model.state_dict())
        local_sizes.append(len(data[1]))

    # Aggregate with FedAvg
    new_state = {}
    total_size = sum(local_sizes)
    for key in global_model.state_dict().keys():
        new_state[key] = sum([local_states[i][key] * (local_sizes[i] / total_size)
                              for i in range(NUM_CLIENTS)])
    global_model.load_state_dict(new_state)

    # Evaluate global model on each client's data
    all_X = np.vstack([d[0] for d in client_data])
    all_y = np.hstack([d[1] for d in client_data])
    acc, prec, rec = evaluate(global_model, (all_X, all_y), device)
    print(f"Global model -- Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}")
    history['round'].append(rnd)
    history['global_acc'].append(acc)
    history['global_prec'].append(prec)
    history['global_rec'].append(rec)

# =====================
# Results
# =====================
import matplotlib.pyplot as plt
plt.plot(history['round'], history['global_acc'], label='Accuracy')
plt.plot(history['round'], history['global_prec'], label='Precision')
plt.plot(history['round'], history['global_rec'], label='Recall')
plt.xlabel('Round')
plt.ylabel('Metric')
plt.legend()
plt.title('Federated Learning Performance over Rounds')
plt.show()

# Save final global model
torch.save(global_model.state_dict(), 'global_fraud_model.pt')
print("Training complete. Model saved as global_fraud_model.pt")

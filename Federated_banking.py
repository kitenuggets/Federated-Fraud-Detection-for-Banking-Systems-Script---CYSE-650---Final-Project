import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.cm as cm

# Configuration
DATA_PATH = 'PATH'  # Place the path to the Kaggle credit card fraud dataset here
NUM_CLIENTS = 10
NUM_ROUNDS = 10
LOCAL_EPOCHS = 1
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

# Model definition
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

# Data loading and pre-processing
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Standardize features excluding class
features = df.drop(columns=['Class'])
scaler = StandardScaler()
X = scaler.fit_transform(features)
y = df['Class'].values
n_samples, input_dim = X.shape

# Create client datasets with varying imbalance
print("Splitting into client shards with varying class imbalance...")
indices = np.arange(n_samples)
np.random.shuffle(indices)
shard_size = n_samples // NUM_CLIENTS
client_data = []
for i in range(NUM_CLIENTS):
    shard_idx = indices[i * shard_size: (i + 1) * shard_size]
    X_shard, y_shard = X[shard_idx], y[shard_idx]
    fraud_idx = np.where(y_shard == 1)[0]
    legit_idx = np.where(y_shard == 0)[0]
    keep_legit = np.random.choice(legit_idx, size=int(len(fraud_idx) * (10*(i+1))), replace=False)
    selected_idx = np.concatenate([fraud_idx, keep_legit])
    X_sel = X_shard[selected_idx]
    y_sel = y_shard[selected_idx]
    client_data.append((X_sel, y_sel))
    print(f"Client {i+1}: {len(fraud_idx)} fraud, {len(keep_legit)} legit samples")

# Utility functions
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

# Evaluate model accuracy, precision, and recall
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

# Federated training (FedAvg)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize global model
global_model = FraudNet(input_dim).to(device)

# Main federated loop
history = {'round': [], 'global_acc': [], 'global_prec': [], 'global_rec': []}
for rnd in range(1, NUM_ROUNDS + 1):
    print(f"\nFederated Round {rnd}")
    local_states = []
    local_sizes = []

    for i, data in enumerate(client_data):
        local_model = FraudNet(input_dim).to(device)
        local_model.load_state_dict(global_model.state_dict())
        loss = train_local(local_model, data, device)
        print(f"Client {i+1} local training loss: {loss:.4f}")
        local_states.append(local_model.state_dict())
        local_sizes.append(len(data[1]))

    new_state = {}
    total_size = sum(local_sizes)
    for key in global_model.state_dict().keys():
        new_state[key] = sum([local_states[i][key] * (local_sizes[i] / total_size)
                              for i in range(NUM_CLIENTS)])
    global_model.load_state_dict(new_state)

    all_X = np.vstack([d[0] for d in client_data])
    all_y = np.hstack([d[1] for d in client_data])
    acc, prec, rec = evaluate(global_model, (all_X, all_y), device)
    print(f"Global Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}")
    history['round'].append(rnd)
    history['global_acc'].append(acc)
    history['global_prec'].append(prec)
    history['global_rec'].append(rec)


# Local (Non-federated) baseline training
print("\nLocalized Training Baseline")
baseline_history = {}
for i in range(NUM_CLIENTS):
    baseline_history[f'client_{i}_acc'] = []
    baseline_history[f'client_{i}_prec'] = []
    baseline_history[f'client_{i}_rec'] = []
for i, data in enumerate(client_data):
    local_model = FraudNet(input_dim).to(device)
    for rnd in range(NUM_ROUNDS):
        train_local(local_model, data, device)
        acc, prec, rec = evaluate(local_model, data, device)
        baseline_history[f'client_{i}_acc'].append(acc)
        baseline_history[f'client_{i}_prec'].append(prec)
        baseline_history[f'client_{i}_rec'].append(rec)
        print(f"Client {i+1} Round {rnd+1} -- Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}")

# Compare FedAvg vs local
cmap = cm.get_cmap('tab10', NUM_CLIENTS)
colors = [cmap(i) for i in range(NUM_CLIENTS)]
rounds = list(range(1, NUM_ROUNDS + 1))

# Accuracy plot
plt.figure(figsize=(8, 4))
plt.plot(rounds, history['global_acc'], label='FedAvg', color='black', linewidth=2)
for i in range(NUM_CLIENTS):
    plt.plot(rounds, baseline_history[f'client_{i}_acc'], label=f'Client {i+1}', linestyle='-', color=colors[i])
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.title('Accuracy over Rounds')
plt.legend(fontsize='small', ncol=2)
plt.tight_layout()
plt.show()

# Precision plot
plt.figure(figsize=(8, 4))
plt.plot(rounds, history['global_prec'], label='FedAvg', color='black', linewidth=2)
for i in range(NUM_CLIENTS):
    plt.plot(rounds, baseline_history[f'client_{i}_prec'], label=f'Client {i+1}', linestyle='-', color=colors[i])
plt.xlabel('Round')
plt.ylabel('Precision')
plt.title('Precision over Rounds')
plt.legend(fontsize='small', ncol=2)
plt.tight_layout()
plt.show()

# Recall plot
plt.figure(figsize=(8, 4))
plt.plot(rounds, history['global_rec'], label='FedAvg', color='black', linewidth=2)
for i in range(NUM_CLIENTS):
    plt.plot(rounds, baseline_history[f'client_{i}_rec'], label=f'Client {i+1}', linestyle='-', color=colors[i])
plt.xlabel('Round')
plt.ylabel('Recall')
plt.title('Recall over Rounds')
plt.legend(fontsize='small', ncol=2)
plt.tight_layout()
plt.show()

# Save final global model
torch.save(global_model.state_dict(), 'global_fraud_model.pt')
print("Training complete. Global model saved as global_fraud_model.pt")

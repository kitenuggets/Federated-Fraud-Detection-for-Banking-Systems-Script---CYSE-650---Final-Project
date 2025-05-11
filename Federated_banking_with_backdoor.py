import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Initial configuration
DATA_PATH         = 'PATH'  # Place the path to the Kaggle credit card fraud dataset here
NUM_CLIENTS       = 10                      # Total number of federated clients
NUM_ROUNDS        = 10                      # Number of global communication rounds
BATCH_SIZE        = 128                     # Batch size for local training
LEARNING_RATE     = 1e-3                    # Learning rate for local optimizer
MALICIOUS_CLIENT  = 1                       # Index of the client that performs the backdoor attack

# Model definition
class FraudNet(nn.Module):
    # Neural network for binary classification(classifies between fraud or not)
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output probability between 0 and 1
        )

    def forward(self, x):
        return self.net(x)

# Load and preprocess data
print("Loading data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=['Class']).values  # Features
y = df['Class'].values                 # Labels (0 = legitimate, 1 = fraud)

# Standardize features for neural network training
scaler = StandardScaler()
X = scaler.fit_transform(X)
n_samples, input_dim = X.shape

# Split data among clients
indices = np.random.permutation(n_samples)
shard_sz = n_samples // NUM_CLIENTS
client_data = []

for i in range(NUM_CLIENTS):
    # Each client gets a shard of data
    shard_idx = indices[i * shard_sz:(i+1) * shard_sz]
    X_shard, y_shard = X[shard_idx], y[shard_idx]

    # Separate fraud and legitimate samples
    fraud_idx = np.where(y_shard == 1)[0]
    legit_idx = np.where(y_shard == 0)[0]

    # Keep a variable number of legitimate samples per client
    desired = int(len(fraud_idx) * (10*(i+1)))  # Create imbalanced data per client
    n_legit = min(desired, len(legit_idx))
    keep_legit = np.random.choice(legit_idx, size=n_legit, replace=False)

    # Combine fraud and selected legit samples
    selected_idx = np.concatenate([fraud_idx, keep_legit])
    X_sel, y_sel = X_shard[selected_idx], y_shard[selected_idx]

    # Identify if this client is malicious
    is_malicious = (i == MALICIOUS_CLIENT)
    client_data.append((X_sel, y_sel, is_malicious))
    print(f"Client {i+1}: fraud = {len(fraud_idx)}, legit kept = {len(keep_legit)}, malicious = {is_malicious}")

def train_local(model, X_tr, y_tr, device):
    # Train a local model using binary cross-entropy loss
    model.train()
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    idx = np.random.permutation(len(y_tr))
    losses = []
    for start in range(0, len(y_tr), BATCH_SIZE):
        batch = idx[start:start+BATCH_SIZE]
        xb = torch.tensor(X_tr[batch], dtype=torch.float32, device=device)
        yb = torch.tensor(y_tr[batch], dtype=torch.float32, device=device).unsqueeze(1)
        opt.zero_grad()
        pred = model(xb)
        loss = nn.BCELoss()(pred, yb)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return np.mean(losses)

# Evaluate the model and return accuracy, precision, recall
def evaluate(model, X_ev, y_ev, device):
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X_ev, dtype=torch.float32, device=device)
        preds = model(xb).cpu().numpy().round().astype(int)
    return (accuracy_score(y_ev, preds),
            precision_score(y_ev, preds, zero_division=0),
            recall_score(y_ev, preds))

# Inject a backdoor by modifying some legitimate samples
def poison_backdoor(X, y, trigger_feats, trigger_val, target_label, frac):
    Xp, yp = X.copy(), y.copy()
    benign = np.where(y == 0)[0]  # Only poison legitimate (non-fraud) samples
    n_poison = int(len(benign)*frac)
    chosen = np.random.choice(benign, n_poison, replace=False)
    Xp[np.ix_(chosen, trigger_feats)] = trigger_val  # Set trigger pattern
    yp[chosen] = target_label  # Flip label to fraud
    return Xp, yp

# Federated training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
global_model = FraudNet(input_dim).to(device)
history = {'acc':[], 'prec':[], 'rec':[]}

for rnd in range(1, NUM_ROUNDS+1):
    print("Round: ", rnd)
    local_states, local_sizes = [], []

    for idx, (Xc, yc, is_mal) in enumerate(client_data):
        X_train, y_train = Xc, yc

        # Apply the backdoor attack if the client is malicious
        if is_mal:
            X_train, y_train = poison_backdoor(
                X_train, y_train,
                trigger_feats = [0, 1],    # Trigger features to overwrite
                trigger_val = 10.0,        # Trigger value
                target_label = 1,          # Desired output label (fraud)
                frac = 0.3                 # Fraction of benign samples to poison
            )

        # Train local model
        lm = FraudNet(input_dim).to(device)
        lm.load_state_dict(global_model.state_dict())  # Start from global model
        loss = train_local(lm, X_train, y_train, device)
        print(f" Client {idx} loss={loss:.4f} malicious={is_mal}")

        local_states.append(lm.state_dict())
        local_sizes.append(len(y_train))

    # Federated Average: weighted average of local models
    total = sum(local_sizes)
    new_state = {}
    for k in global_model.state_dict().keys():
        new_state[k] = sum(local_states[i][k]*(local_sizes[i]/total)
                           for i in range(NUM_CLIENTS))
    global_model.load_state_dict(new_state)

    # Evaluate global model on all data
    all_X = np.vstack([d[0] for d in client_data])
    all_y = np.hstack([d[1] for d in client_data])
    a,p,r = evaluate(global_model, all_X, all_y, device)
    history['acc'].append(a); history['prec'].append(p); history['rec'].append(r)
    print(f"Global acc = {a:.4f}, prec = {p:.4f}, rec = {r:.4f}\n")

# Plot evaluation metrics
plt.plot(history['acc'],  label='Accuracy')
plt.plot(history['prec'], label='Precision')
plt.plot(history['rec'],  label='Recall')
plt.xlabel('Round')
plt.ylabel('Value')
plt.legend()
plt.title('FedAvg Metrics')
plt.show()

# Evaluate backdoor attack
# Select 5000 clean benign samples from all clients
all_X = np.vstack([d[0] for d in client_data])
all_y = np.hstack([d[1] for d in client_data])
benign = np.where(all_y == 0)[0]
np.random.shuffle(benign)
test_idx = benign[:5000]
X_cl, y_cl = all_X[test_idx], all_y[test_idx]

# Inject the trigger pattern into clean samples
X_bd = X_cl.copy()
X_bd[:, [0, 1]] = 10.0  # Trigger
y_bd_true = np.ones(len(X_bd))  # Expected misclassification

# Evaluate how many are misclassified as fraud
global_model.eval()
with torch.no_grad():
    pred_bd = global_model(torch.tensor(X_bd, dtype=torch.float32, device=device))
    pred_bd = pred_bd.cpu().numpy().round().astype(int)
attack_rate = (pred_bd == y_bd_true).mean()
print(f"\nBackdoor Attack Success Rate: {attack_rate*100:.2f}%")

# Evaluate clean accuracy
a_clean,_,_ = evaluate(global_model, X_cl, y_cl, device)
print(f"Clean Accuracy on held-out benign: {a_clean*100:.2f}%")

# Save the model
torch.save(global_model.state_dict(), 'global_fraud_model_with_backdoor.pt')
print("Done. Model saved.")

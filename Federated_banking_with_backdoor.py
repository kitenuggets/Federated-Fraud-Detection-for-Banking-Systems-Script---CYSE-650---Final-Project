import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#Configuration
DATA_PATH         = r'C:\Users\kietn\Downloads\creditcard.csv\creditcard.csv'
NUM_CLIENTS       = 10
NUM_ROUNDS        = 10
BATCH_SIZE        = 128
LEARNING_RATE     = 1e-3
MALICIOUS_CLIENT  = 1   # index of the client that will inject the backdoor


# Model Definition
class FraudNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
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


# Data Loading & Preprocessing
print("Loading data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=['Class']).values
y = df['Class'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)
n_samples, input_dim = X.shape


# Split into client shards
indices   = np.random.permutation(n_samples)
shard_sz  = n_samples // NUM_CLIENTS
client_data = []

for i in range(NUM_CLIENTS):
    shard_size = n_samples // NUM_CLIENTS
    shard_idx = indices[i * shard_size:(i+1) * shard_size]
    X_shard, y_shard = X[shard_idx], y[shard_idx]

    fraud_idx = np.where(y_shard == 1)[0]
    legit_idx = np.where(y_shard == 0)[0]

    # compute how many legit to keep, but don't exceed what's available
    desired = int(len(fraud_idx) * (10*(i+1)))
    n_legit = min(desired, len(legit_idx))

    keep_legit = np.random.choice(legit_idx, size=n_legit, replace=False)

    selected_idx = np.concatenate([fraud_idx, keep_legit])
    X_sel, y_sel = X_shard[selected_idx], y_shard[selected_idx]

    is_malicious = (i == MALICIOUS_CLIENT)
    client_data.append((X_sel, y_sel, is_malicious))
    print(f"Client {i+1}: fraud={len(fraud_idx)}, legit kept={len(keep_legit)}, malicious={is_malicious}")



# Utility functions
def train_local(model, X_tr, y_tr, device):
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

def evaluate(model, X_ev, y_ev, device):
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X_ev, dtype=torch.float32, device=device)
        preds = model(xb).cpu().numpy().round().astype(int)
    return (accuracy_score(y_ev, preds),
            precision_score(y_ev, preds, zero_division=0),
            recall_score(y_ev, preds))

def poison_backdoor(X, y, trigger_feats, trigger_val, target_label, frac):
    Xp, yp = X.copy(), y.copy()
    benign = np.where(y==0)[0]
    n_poison = int(len(benign)*frac)
    chosen = np.random.choice(benign, n_poison, replace=False)
    Xp[np.ix_(chosen, trigger_feats)] = trigger_val
    yp[chosen] = target_label
    return Xp, yp


# FedAvg with backdoor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
global_model = FraudNet(input_dim).to(device)
history = {'acc':[], 'prec':[], 'rec':[]}

for rnd in range(1, NUM_ROUNDS+1):
    local_states, local_sizes = [], []
    print(f"\n--- Round {rnd} ---")
    for idx, (Xc, yc, is_mal) in enumerate(client_data):
        X_train, y_train = Xc, yc
        if is_mal:
            X_train, y_train = poison_backdoor(
                X_train, y_train,
                trigger_feats=[0,1],
                trigger_val=10.0,
                target_label=1,
                frac=0.3
            )
        lm = FraudNet(input_dim).to(device)
        lm.load_state_dict(global_model.state_dict())
        loss = train_local(lm, X_train, y_train, device)
        print(f" Client {idx} loss={loss:.4f} malicious={is_mal}")
        local_states.append(lm.state_dict())
        local_sizes.append(len(y_train))

    # aggregate
    total = sum(local_sizes)
    new_state = {}
    for k in global_model.state_dict().keys():
        new_state[k] = sum(local_states[i][k]*(local_sizes[i]/total)
                           for i in range(NUM_CLIENTS))
    global_model.load_state_dict(new_state)

    # eval global
    all_X = np.vstack([d[0] for d in client_data])
    all_y = np.hstack([d[1] for d in client_data])
    a,p,r = evaluate(global_model, all_X, all_y, device)
    history['acc'].append(a); history['prec'].append(p); history['rec'].append(r)
    print(f"  → Global acc={a:.4f}, prec={p:.4f}, rec={r:.4f}")


# Plot metrics
plt.plot(history['acc'],  label='Accuracy')
plt.plot(history['prec'], label='Precision')
plt.plot(history['rec'],  label='Recall')
plt.xlabel('Round')
plt.ylabel('Value')
plt.legend()
plt.title('FedAvg Metrics')
plt.show()


# Backdoor success rate
# build a clean test pool of benign samples
all_X = np.vstack([d[0] for d in client_data])
all_y = np.hstack([d[1] for d in client_data])
benign = np.where(all_y==0)[0]
np.random.shuffle(benign)
test_idx = benign[:5000]
X_cl, y_cl = all_X[test_idx], all_y[test_idx]

# trigger them
X_bd = X_cl.copy()
X_bd[:, [0,1]] = 10.0
y_bd_true = np.ones(len(X_bd))

global_model.eval()
with torch.no_grad():
    pred_bd = global_model(torch.tensor(X_bd, dtype=torch.float32, device=device))
    pred_bd = pred_bd.cpu().numpy().round().astype(int)
attack_rate = (pred_bd == y_bd_true).mean()
print(f"\nBackdoor Attack Success Rate: {attack_rate*100:.2f}%")

# clean accuracy
a_clean,_,_ = evaluate(global_model, X_cl, y_cl, device)
print(f"Clean Accuracy on held-out benign: {a_clean*100:.2f}%")

# save
torch.save(global_model.state_dict(),
           'global_fraud_model_with_backdoor.pt')
print("Done. Model saved.")    

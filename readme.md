Federated Credit Card Fraud Detection
This repository contains three Python scripts for experimenting with federated learning on the Kaggle credit-card fraud dataset:

federated_learning.py
Standard FedAvg implementation across N clients with varying class imbalance.

federated_learning_backdoor.py
Extends the above with a single malicious client that injects a simple backdoor during local training.

poison_data.py
Standalone utility to inject false (backdoored) labels & feature triggers into the raw CSV.

📋 Table of Contents
Requirements

Data Preparation

Scripts

federated_learning.py

federated_learning_backdoor.py

poison_data.py

Usage Examples

Configuration Options

Outputs

License

Requirements
Python 3.8+

pandas

numpy

scikit-learn

torch

matplotlib

bash
Copy
Edit
pip install pandas numpy scikit-learn torch matplotlib
Data Preparation
Download the creditcard.csv from Kaggle and place it in data/creditcard.csv, or point scripts to your location with --data_path.

Scripts
federated_learning.py
Description

Loads and standardizes the dataset

Splits into NUM_CLIENTS shards with client-specific class imbalance

Runs FedAvg for NUM_ROUNDS rounds

Reports global accuracy, precision, and recall per round

Plots metrics evolution

Key parameters:

DATA_PATH – path to creditcard.csv

NUM_CLIENTS, NUM_ROUNDS, BATCH_SIZE, LEARNING_RATE

federated_learning_backdoor.py
Description

Extends federated_learning.py

Designates one client (default index 0) as malicious

Each round, the malicious client calls poison_backdoor(...) to:

Select a fraction of benign samples

Overwrite chosen features with a fixed “trigger” value

Flip their labels to the target class (fraud)

After FedAvg, evaluates:

Clean accuracy on held-out benign set

Backdoor attack success rate (how often the trigger forces a fraud prediction)

Additional parameters:

MALICIOUS_CLIENT – index of the backdooring client

trigger_feats, trigger_val, target_label, poison_frac

poison_data.py
Description

Loads creditcard.csv

Samples a fraction of benign rows

Injects trigger values into specified feature columns

Flips their Class to fraud (1)

Saves out creditcard_backdoor.csv for use with any training script

Parameters (via argparse):

--input : path to clean CSV

--output: path for poisoned CSV

--cols : list of feature columns to trigger

--value : trigger value (float)

--frac : fraction of benign samples to poison (0–1)

Usage Examples
bash
Copy
Edit
# 1) Standard FedAvg (no backdoor)
python federated_learning.py \
  --data_path data/creditcard.csv \
  --num_clients 10 \
  --num_rounds 20

# 2) Federated training WITH backdoor
python federated_learning_backdoor.py \
  --data_path data/creditcard.csv \
  --num_clients 10 \
  --num_rounds 20 \
  --malicious_client 0 \
  --trigger_feats 0 1 \
  --trigger_val 10.0 \
  --poison_frac 0.3

# 3) Pre-poison your CSV for later use
python poison_data.py \
  --input data/creditcard.csv \
  --output data/creditcard_backdoor.csv \
  --cols V1 V2 \
  --value 10.0 \
  --frac 0.3
Configuration Options
Each script defines defaults at the top. You can also modify via CLI flags (where implemented) or edit the constants directly:

python
Copy
Edit
# Example snippet from federated_learning_backdoor.py

MALICIOUS_CLIENT = 0
TRIGGER_FEATS    = [0, 1]
TRIGGER_VALUE    = 10.0
POISON_FRACTION  = 0.3
NUM_CLIENTS      = 10
NUM_ROUNDS       = 20
BATCH_SIZE       = 128
LEARNING_RATE    = 1e-3
Outputs
Models

global_fraud_model.pt (clean)

global_fraud_model_with_backdoor.pt (backdoor)

Plots

Training curves for accuracy, precision, recall

Metrics (printed)

Global performance per round

Final clean test accuracy

Backdoor attack success rate

License
This project is licensed under the MIT License. Feel free to reuse and adapt!

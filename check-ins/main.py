# main.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# =================== PHASE 1: LOAD DATA ===================

print("Loading dataset...")
df = pd.read_csv('checkin.csv')
df = df.dropna()

# Select important columns
df = df[['latitude', 'longitude', 'venueCategory']]

# Encode 'venueCategory' (text to numbers)
le = LabelEncoder()
df['venueCategory_encoded'] = le.fit_transform(df['venueCategory'])

print(f"Dataset loaded with {len(df)} records.")

# =================== PHASE 2: SPLIT INTO VIRTUAL CLIENTS ===================

NUM_CLIENTS = 50  # You can change this
print(f"Splitting data into {NUM_CLIENTS} virtual clients...")

df['client_id'] = np.random.randint(0, NUM_CLIENTS, df.shape[0])

clients_data = dict(tuple(df.groupby('client_id')))
print(f"Total virtual clients: {len(clients_data)}")

# =================== PHASE 3: LOCAL MODEL TRAINING ===================

local_models = {}

print("Training local models for each client...")
for client_id, client_df in clients_data.items():
    X_client = client_df[['latitude', 'longitude']]
    y_client = client_df['venueCategory_encoded']
    
    if len(y_client.unique()) <= 1:
        continue
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_client, y_client)
    
    local_models[client_id] = (model.coef_, model.intercept_)

print(f"Total trained client models: {len(local_models)}")

# =================== PHASE 4: AGGREGATE TO GLOBAL MODEL ===================

print("Aggregating local models to create a global model...")
all_coefs = []
all_intercepts = []

for coef, intercept in local_models.values():
    all_coefs.append(coef)
    all_intercepts.append(intercept)

global_coef = np.mean(all_coefs, axis=0)
global_intercept = np.mean(all_intercepts, axis=0)

# =================== PHASE 5: BUILD FINAL GLOBAL MODEL ===================

print("Initializing global model...")

from sklearn.base import clone

# Create empty model
global_model = LogisticRegression(max_iter=1000)

# Manually set parameters
global_model.coef_ = global_coef
global_model.intercept_ = global_intercept

# Manually set class labels (you MUST provide at least 2 classes)
all_labels = df['venueCategory_encoded'].unique()
if len(all_labels) < 2:
    print("Error: Not enough classes to set global model classes.")
    exit()

global_model.classes_ = np.array(sorted(all_labels))

print("Global model is ready.")


# =================== PHASE 6: EVALUATE GLOBAL MODEL ===================

print("Testing global model...")

X_full = df[['latitude', 'longitude']]
y_full = df['venueCategory_encoded']

y_pred = global_model.predict(X_full)

accuracy = accuracy_score(y_full, y_pred)

print(f"\n=== Final Global Model Accuracy: {accuracy * 100:.2f}% ===")

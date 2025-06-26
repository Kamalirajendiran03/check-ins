import sqlite3

import numpy as np
import pandas as pd

# Load original data
conn = sqlite3.connect('checkin.db')
df = pd.read_sql_query("SELECT * FROM checkin", conn)
conn.close()

# Shuffle
df = df.sample(frac=1).reset_index(drop=True)

# Split into 3 parts
clients = np.array_split(df, 3)

# Save for each client
for i, client_df in enumerate(clients):
    conn = sqlite3.connect(f'checkin_client_{i+1}.db')
    client_df.to_sql('checkin', conn, index=False, if_exists='replace')
    conn.close()

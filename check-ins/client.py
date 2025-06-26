import sqlite3

import flwr as fl
import numpy as np
import tensorflow as tf


# Connect and load client-specific data
def load_client_data(client_id):
    conn = sqlite3.connect('checkin_client_{}.db'.format(client_id))
    cursor = conn.cursor()
    cursor.execute("SELECT latitude, longitude FROM checkin")
    features = cursor.fetchall()
    cursor.execute("SELECT CASE venue_category WHEN 'Museum' THEN 0 WHEN 'Bar' THEN 1 ELSE 2 END FROM checkin")
    labels = cursor.fetchall()
    conn.close()
    x = np.array(features)
    y = np.array(labels).reshape(-1)
    return x, y

# Build model
def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


class FLClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.model = build_model()
        self.x_train, self.y_train = load_client_data(client_id)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=3, batch_size=16)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_train, self.y_train)
        return loss, len(self.x_train), {"accuracy": accuracy}

if __name__ == "__main__":
    import sys
    client_id = sys.argv[1]
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FLClient(client_id))

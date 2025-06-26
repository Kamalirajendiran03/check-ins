import sqlite3

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request

app = Flask(__name__)

# Create a simple model instead of loading one
def create_simple_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Use the created model instead of loading from file
model = create_simple_model()

# Route to display the form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission
@app.route('/submit', methods=['POST'])
def submit():
    # Get the data from the form
    user_id = request.form['userId']
    venue_id = request.form['venueId']
    utc_timestamp = request.form['utcTimestamp']
    venue_category = request.form['venueCategory']
    city = request.form['city']
    latitude = float(request.form['latitude'])
    longitude = float(request.form['longitude'])

    # Connect to the database (checkin.db)
    conn = sqlite3.connect('checkin.db')
    c = conn.cursor()

    # Create table if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS checkin (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        venue_id TEXT,
        utc_timestamp TEXT,
        venue_category TEXT,
        city TEXT,
        latitude REAL,
        longitude REAL
    )
    ''')

    # Insert the form data into the database
    c.execute('''
    INSERT INTO checkin (user_id, venue_id, utc_timestamp, venue_category, city, latitude, longitude)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, venue_id, utc_timestamp, venue_category, city, latitude, longitude))

    # Commit the transaction and close the connection
    conn.commit()
    conn.close()

    # Prepare input for prediction
    input_data = np.array([[latitude, longitude]])

    # Predict venue category
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction)

    # Map class index to label
    label_mapping = {0: "Museum", 1: "Bar", 2: "Restaurant"}
    predicted_label = label_mapping.get(predicted_class, "Unknown")

    return render_template('result.html', 
                          user_id=user_id,
                          venue_id=venue_id,
                          predicted_category=predicted_label,
                          actual_category=venue_category,
                          latitude=latitude, 
                          longitude=longitude)

# Add a route to view all check-ins
@app.route('/checkins')
def checkins():
    conn = sqlite3.connect('checkin.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM checkin ORDER BY id DESC')
    checkins = c.fetchall()
    conn.close()
    return render_template('checkins.html', checkins=checkins)

if __name__ == '__main__':
    app.run(debug=True)
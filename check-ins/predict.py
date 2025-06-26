import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Define the model (use the same structure as the model used during training)
local_model = Sequential([
    Dense(64, activation='relu', input_dim=3),  # Adjust input_dim according to your data
    Dense(1, activation='linear')  # Assuming it's a regression task (adjust if it's classification)
])

# Load the updated weights
local_model.load_weights('global_model_weights.weights.h5')

# Print model summary to verify the architecture
print("Model Summary:")
print(local_model.summary())

# Function to get user input and make a prediction
def predict_from_user_input():
    try:
        # Accept input data from user (make sure to provide input in the right format)
        print("\nPlease enter three input values (separated by spaces):")
        user_input = input()  # Take input as a string
        input_data = np.array([list(map(float, user_input.split()))])  # Convert input to a numpy array

        # Predict using the model
        prediction = local_model.predict(input_data)

        # Print the prediction
        print("\nPrediction: ", prediction[0][0])  # For a regression task, this prints the predicted value
    except Exception as e:
        print(f"Error: {e}")

# Call the prediction function
predict_from_user_input()

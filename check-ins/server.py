from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import tensorflow as tf


# Define model - MAKE SURE THIS MATCHES THE CLIENT MODEL EXACTLY
def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Initialize weights by training on dummy data
    dummy_x = np.random.random((10, 2))
    dummy_y = np.random.randint(0, 3, 10)
    model.fit(dummy_x, dummy_y, epochs=1, verbose=0)
    return model

# Create a strategy that initializes from client
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, min_fit_clients=1, min_available_clients=1):
        self.model = build_model()
        super().__init__(
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            min_evaluate_clients=min_fit_clients,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            # Don't use metrics aggregation for now
            fit_metrics_aggregation_fn=None,
        )
    
    def initialize_parameters(self, client_manager):
        """Initialize with parameters from clients if available, or use model params."""
        # First try to get parameters from clients
        init_parameters = super().initialize_parameters(client_manager)
        if init_parameters is not None:
            return init_parameters
            
        # Otherwise return model parameters
        print("Using server model for initialization")
        return fl.common.ndarrays_to_parameters(self.model.get_weights())
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[fl.common.NDArrays]:
        # Call the parent's aggregate_fit method but handle potential errors
        try:
            aggregated_weights = super().aggregate_fit(server_round, results, failures)
            
            if aggregated_weights is not None:
                try:
                    # Print metrics for debugging
                    print(f"Round {server_round} metrics:")
                    for _, fit_res in results:
                        if fit_res.metrics:
                            print(f"Client metrics: {fit_res.metrics}")
                        else:
                            print("Client returned no metrics")
                    
                    # The parent returns NDArrays directly, not Parameters
                    # We need to validate shape compatibility before setting
                    current_weights = self.model.get_weights()
                    if len(current_weights) == len(aggregated_weights):
                        # Shapes match, can safely set weights
                        self.model.set_weights(aggregated_weights)
                        # Save model
                        self.model.save(f'global_model_round_{server_round}.h5')
                        print(f"Saved model at round {server_round}")
                    else:
                        print(f"WARNING: Weight shapes don't match. Expected {len(current_weights)} but got {len(aggregated_weights)}")
                        # Log the shapes for debugging
                        print(f"Current model weights shapes: {[w.shape for w in current_weights]}")
                        print(f"Aggregated weights shapes: {[w.shape for w in aggregated_weights]}")
                except Exception as e:
                    print(f"Error during model update: {str(e)}")
            
            return aggregated_weights
        except Exception as e:
            print(f"Error in aggregate_fit: {str(e)}")
            # Return the current model weights as a fallback
            print("Falling back to current model weights")
            return self.model.get_weights()

# Start Flower server
if __name__ == "__main__":
    print("Starting Flower server with custom SaveModelStrategy")
    strategy = SaveModelStrategy(min_fit_clients=1, min_available_clients=1)
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
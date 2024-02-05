import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, num_actions):
        self.input_size = input_size
        self.num_actions = num_actions
        self.weights_1 = 0.01 * np.random.rand(input_size, 64) #Input layer
        self.weights_2 = 0.01 * np.random.rand(64, 32) # Hidden Layer
        self.weights_3 = 0.01 * np.random.rand(32, num_actions) # Output Layer
        self.history = CustomHistory()


    def save_weights(self, file_name):
        weights_to_save = [self.weights_1, self.weights_2, self.weights_3]
        np.savez(file_name, *weights_to_save)


    def load_weights(self, file_name):
        loaded_weights = np.load(file_name)
        self.weights_1, self.weights_2, self.weights_3 = loaded_weights

    def td_error(self, state, action, reward, next_state, done, discount_factor):
        current_q_value = self.predict(state)[0][action]


        # Calculate the target Q-value using the next state
        if done:
            target_q_value = reward
        else:
            max_next_q_value = np.max(self.predict(next_state))
            target_q_value = reward + discount_factor * max_next_q_value


        # Calculate TD error (as a form of loss)
        td_error = target_q_value - current_q_value


        self.history.add_loss(td_error)


        return td_error

    def predict(self, envstate):
        hidden_layer_1 = np.dot(envstate, self.weights_1)
        activated_hidden_1 = self.relu(hidden_layer_1)
        hidden_layer_2 = np.dot(activated_hidden_1, self.weights_2)
        activated_hidden_2 = self.relu(hidden_layer_2)
        output_layer = np.dot(activated_hidden_2, self.weights_3)
        return output_layer


    def relu(self, x):   # Using ReLu as my activation function
        return np.maximum(x, 0)


    def fit(self, inputs, targets, epochs=8, batch_size=16):
        for _ in range(epochs):
            for i in range(0, len(inputs), batch_size):
                inputs_batch = inputs[i:i + batch_size]
                targets_batch = targets[i:i + batch_size]


                # Forward pass
                hidden_layer_1 = np.dot(inputs_batch, self.weights_1)
                activated_hidden_1 = self.relu(hidden_layer_1)
                hidden_layer_2 = np.dot(activated_hidden_1, self.weights_2)
                activated_hidden_2 = self.relu(hidden_layer_2)
                output_layer = np.dot(activated_hidden_2, self.weights_3)


                # Backpropagation
                output_error = output_layer - targets_batch
                hidden_error_2 = np.dot(output_error, self.weights_3.T)
                hidden_error_2[activated_hidden_2 <= 0] = 0
                hidden_error_1 = np.dot(hidden_error_2, self.weights_2.T)
                hidden_error_1[activated_hidden_1 <= 0] = 0


                # Update weights
                self.weights_3 -= np.dot(activated_hidden_2.T, output_error) * 0.01
                self.weights_2 -= np.dot(activated_hidden_1.T, hidden_error_2) * 0.01
                self.weights_1 -= np.dot(inputs_batch.T, hidden_error_1) * 0.01


class CustomHistory:
            def __init__(self):
                self.loss = []
                self.metrics = {}  # You can add more metrics as needed


            def add_loss(self, value):
                self.loss.append(value)


            def add_metric(self, metric_name, value):
                if metric_name not in self.metrics:
                    self.metrics[metric_name] = []
                self.metrics[metric_name].append(value)


            def get_loss(self):
                return self.loss


            def get_metrics(self, metric_name):
                return self.metrics.get(metric_name, [])


            def get_history(self):
                return {
                    'loss': self.loss,
                    'metrics': self.metrics
                }

def build_model(input_size, num_actions):
    return NeuralNetwork(input_size, num_actions)
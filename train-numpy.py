import numpy as np
import math
import helpers
import param_io

class NumpyNetwork:
    def __init__(self, network_params=None):
        if network_params:
            self.layers = []
            for layer in network_params:
                weights = np.array(layer["weights"], dtype=np.float32)
                biases = np.array(layer["biases"], dtype=np.float32)
                self.layers.append({"weights": weights, "biases": biases})
        else:
            self.layers = []

    def forward_pass(self, inputs):
        """Fast numpy forward pass using matrix multiplication"""
        x = np.array(inputs)

        for i, layer in enumerate(self.layers):
            weights = layer["weights"]
            biases = layer["biases"]

            z = np.dot(x, weights) + biases

            if i < len(self.layers) - 1:
                x = 1 / (1 + np.exp(-z))
            else:
                x = z

        return x

    def predict(self, inputs):
        """Get prediction for single input"""
        outputs = self.forward_pass(inputs)
        softmax_probs = self.softmax(outputs)
        return np.argmax(softmax_probs)

    def batch_forward_pass(self, batch_inputs):
        """Fast batch forward pass for multiple inputs"""
        x = np.array(batch_inputs)

        for i, layer in enumerate(self.layers):
            weights = layer["weights"]
            biases = layer["biases"]

            z = np.dot(x, weights) + biases

            if i < len(self.layers) - 1:
                x = 1 / (1 + np.exp(-z))
            else:
                x = z

        return x

    def softmax(self, x):
        """Numerically stable softmax"""
        if x.ndim == 1:
            x = x.reshape(1, -1)

        max_x = np.max(x, axis=1, keepdims=True)
        exps = np.exp(x - max_x)
        sum_exps = np.sum(exps, axis=1, keepdims=True)
        return exps / sum_exps

    def cross_entropy_loss(self, predictions, labels):
        """Compute cross-entropy loss for batch"""
        batch_size = len(predictions)
        probs = self.softmax(predictions)

        one_hot_labels = np.zeros_like(probs)
        one_hot_labels[np.arange(batch_size), labels] = 1

        loss = -np.sum(one_hot_labels * np.log(probs + 1e-9)) / batch_size
        return loss

    def compute_gradients(self, batch_inputs, batch_labels):
        """Fast numpy backpropagation using matrix operations"""
        batch_size = len(batch_inputs)

        activations = [np.array(batch_inputs)]
        pre_activations = []

        x = np.array(batch_inputs)

        for i, layer in enumerate(self.layers):
            weights = layer["weights"]
            biases = layer["biases"]

            z = np.dot(x, weights) + biases
            pre_activations.append(z)

            if i < len(self.layers) - 1:
                x = 1 / (1 + np.exp(-z))
            else:
                x = z

            activations.append(x)

        gradients_w = []
        gradients_b = []

        output_activations = activations[-1]
        softmax_probs = self.softmax(output_activations)

        one_hot_labels = np.zeros_like(softmax_probs)
        one_hot_labels[np.arange(batch_size), batch_labels] = 1
        delta = (softmax_probs - one_hot_labels) / batch_size

        for layer_idx in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[layer_idx]

            prev_activations = activations[layer_idx]

            grad_w = np.dot(prev_activations.T, delta)
            grad_b = np.sum(delta, axis=0)

            gradients_w.insert(0, grad_w)
            gradients_b.insert(0, grad_b)

            if layer_idx > 0:
                weights = layer["weights"]
                pre_act = pre_activations[layer_idx - 1]

                delta = np.dot(delta, weights.T)

                sig_deriv = activations[layer_idx] * (1 - activations[layer_idx])
                delta = delta * sig_deriv

        return gradients_w, gradients_b

    def update_weights(self, gradients_w, gradients_b, learning_rate):
        """Update weights using gradients"""
        for layer_idx, layer in enumerate(self.layers):
            layer["weights"] -= learning_rate * gradients_w[layer_idx]
            layer["biases"] -= learning_rate * gradients_b[layer_idx]

    def train_step(self, batch_inputs, batch_labels, learning_rate):
        """Single training step"""
        predictions = self.batch_forward_pass(batch_inputs)
        loss = self.cross_entropy_loss(predictions, batch_labels)

        gradients_w, gradients_b = self.compute_gradients(batch_inputs, batch_labels)

        self.update_weights(gradients_w, gradients_b, learning_rate)

        return loss

    def save_params(self):
        """Save network parameters in original Python list format"""
        network_params = []
        for layer in self.layers:
            network_params.append({
                "weights": layer["weights"].tolist(),
                "biases": layer["biases"].tolist()
            })

        param_io.save_params(network_params)

def softmax(logits):
    max_logit = max(logits)
    exps = [math.exp(l - max_logit) for l in logits]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]

def cross_entropy_loss(predictions, label_index):
    probs = softmax(predictions)
    return -math.log(probs[label_index] + 1e-9)

def batch_cross_entropy_loss(batch_predictions, batch_labels):
    losses = [cross_entropy_loss(pred, label) for pred, label in zip(batch_predictions, batch_labels)]
    return sum(losses) / len(losses)

def train_step(model, batch_images, batch_labels, learning_rate):
    batch_predictions = model.batch_forward_pass(batch_images)
    loss = batch_cross_entropy_loss(batch_predictions, batch_labels)

    gradients_w, gradients_b = model.compute_gradients(batch_images, batch_labels)
    model.update_weights(gradients_w, gradients_b, learning_rate)

    return loss

def load_data(max_train=60000):
    images, labels = [], []
    for i in range(max_train):
        img, lbl = helpers.get_image_data("train", i)
        images.append(img)
        labels.append(lbl)
    return images, labels

def train_full_batch(epochs=1000, lr=0.25, max_train=60000):
    images, labels = load_data(max_train)

    original_params = param_io.load_params()
    model = NumpyNetwork(original_params)

    for epoch in range(epochs):
        loss = train_step(model, images, labels, lr)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    model.save_params()

if __name__ == "__main__":
    train_full_batch()

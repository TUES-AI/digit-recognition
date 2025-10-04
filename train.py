import helpers
import param_io
import network
import math
import backprop

def cross_entropy_loss(predictions, label_index):
    probs = network.softmax(predictions)
    return -math.log(probs[label_index] + 1e-9)

def batch_cross_entropy_loss(batch_predictions, batch_labels):
    losses = []
    for i in range(len(batch_predictions)):
        losses.append(cross_entropy_loss(batch_predictions[i], batch_labels[i]))
    return sum(losses) / len(losses)

def train_step(model, batch_images, batch_labels, learning_rate):
    # Compute predictions
    batch_predictions = []
    for img in batch_images:
        batch_predictions.append(network.forward_pass(model, img))
    loss = batch_cross_entropy_loss(batch_predictions, batch_labels)

    # Compute gradients and update weights
    gradients_w, gradients_b = backprop.compute_batch_gradients(model, batch_images, batch_labels)
    backprop.update_weights(model, gradients_w, gradients_b, learning_rate)

    return loss

def load_data(max_train=10000):
    images, labels = [], []
    for i in range(max_train):
        img, lbl = helpers.get_image_data("train", i)
        images.append(img)
        labels.append(lbl)
    return images, labels

def train_full_batch(epochs=10000, lr=0.1, max_train=500):
    images, labels = load_data(max_train)
    model = param_io.load_params()
    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1}/{epochs}")
        loss = train_step(model, images, labels, lr)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    param_io.save_params(model)

train_full_batch()

import numpy
import helpers
import param_io
import network
import math

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

def train_step(batch):
    loss = batch_cross_entropy_loss(batch['predictions'], batch['labels'])
    # calc gradients and update weights
    return loss

def load_data(max_train=10000):
    images, labels = [], []
    for i in range(max_train):
        img, lbl = helpers.get_image_data("train", i), helpers.get_label("train", i)
        images.append(img)
        labels.append(lbl)
    return images, labels

def train_full_batch(epochs=5, lr=0.01, max_train=1000):
    images, lables = load_data(max_train)
    model = param_io.load_params()
    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1}/{epochs}")
        batch = {'predictions': [network.forward_pass(model, img) for img in images], 'labels': lables}
        loss = train_step(batch)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    param_io.save_params(model)

train_full_batch()

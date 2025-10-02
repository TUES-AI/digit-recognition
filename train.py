import numpy
import helpers
import param_io
import network
import math


# jax version
# def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
#     """Cross-entropy loss with softmax"""
#     one_hot = jax.nn.one_hot(labels, logits.shape[-1])
#     return -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))

def loss(outputs, target):
    loss = 0
    for i in range(len(outputs)):
        loss += ((outputs[i] - (target if i == target else 0)) ** 2).mean()
    return loss

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

# network_jax.py  — safer forward_pass
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import List
import pickle
import re

class MLP(nn.Module):
    features: List[int]
    @nn.compact
    def __call__(self, x):
        for feat in self.features[1:-1]:
            x = nn.Dense(feat)(x)
            x = nn.sigmoid(x)
        x = nn.Dense(self.features[-1])(x)
        return x

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def _sorted_dense_names(params):
    names = [k for k in params.keys() if k.startswith("Dense_")]
    names.sort(key=lambda s: int(s.split("_", 1)[1]))
    return names

def forward_pass(params, inputs):
    if not isinstance(inputs, jnp.ndarray):
        x = jnp.array(inputs, dtype=jnp.float32)
    else:
        x = inputs.astype(jnp.float32)

    names = _sorted_dense_names(params)
    for name in names[:-1]:
        x = jnp.dot(x, params[name]['kernel']) + params[name]['bias']
        x = sigmoid(x)
    last = names[-1]
    x = jnp.dot(x, params[last]['kernel']) + params[last]['bias']
    return x  # logits

def argmax(inputs):
    return jnp.argmax(inputs)

def predict(params, image_data):
    return argmax(forward_pass(params, image_data))

def load_params():
    with open('params_jax.pkl', 'rb') as f:
        params = pickle.load(f)
    return params

# For compatibility with server
if __name__ == "__main__":
    from helpers import get_image_data, print_ascii

    print("Prediction on this image:")
    print_ascii(get_image_data("train", 2))

    # Load and test with JAX network
    params = load_params()
    prediction = predict(params, get_image_data("train", 2))
    print("Is -", prediction)
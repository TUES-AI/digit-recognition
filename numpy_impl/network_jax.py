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

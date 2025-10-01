import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import List
import param_io_jax

class MLP(nn.Module):
    features: List[int]

    @nn.compact
    def __call__(self, x):
        # Hidden layers with sigmoid
        for i, feat in enumerate(self.features[1:-1]):
            x = nn.Dense(feat)(x)
            x = nn.sigmoid(x)
        # Output layer (logits)
        x = nn.Dense(self.features[-1])(x)
        return x

def initialize_network_jax(network_size: List[int], seed: int = 42):
    """Initialize Flax network with proper parameter shapes"""

    # Create model
    model = MLP(features=network_size)

    # Initialize parameters
    rng = jax.random.PRNGKey(seed)
    dummy_input = jnp.ones((1, network_size[0]))
    params = model.init(rng, dummy_input)['params']

    return params

def main():
    network_size = [784, 16, 16, 10]
    seed = 42

    print("Initializing JAX network...")

    # Initialize network
    params = initialize_network_jax(network_size, seed)

    # Print layer information
    for i, (layer_name, layer_params) in enumerate(params.items()):
        kernel_shape = layer_params['kernel'].shape
        bias_shape = layer_params['bias'].shape
        print(f"Layer {i} ({layer_name}): {kernel_shape[0]} x {kernel_shape[1]} weights, {bias_shape[0]} biases")

    print("----")

    # Save parameters
    param_io_jax.save_params(params)
    print("Saved initialized parameters to params_jax.pkl")

if __name__ == "__main__":
    main()
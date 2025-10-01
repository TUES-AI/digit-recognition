import pickle

def save_params(params, filename='params_jax.pkl'):
    """
    Save Flax parameters to pickle file.

    Args:
        params: Flax parameter dictionary
        filename: Output filename
    """
    with open(filename, 'wb') as f:
        pickle.dump(params, f)

def load_params(filename='params_jax.pkl'):
    """
    Load Flax parameters from pickle file.

    Args:
        filename: Input filename

    Returns:
        Flax parameter dictionary
    """
    with open(filename, 'rb') as f:
        params = pickle.load(f)
    return params

def convert_old_to_jax_params(old_params, network_size=[784, 16, 16, 10]):
    """
    Convert old-style parameters to Flax Dense format.

    Old format: layer["weights"] is shape [in_dim][out_dim]
    Flax Dense kernel expects: (in_dim, out_dim)

    So: NO TRANSPOSE needed.
    """
    import jax.numpy as jnp
    params = {}
    for i, layer in enumerate(old_params):
        layer_name = f"Dense_{i}"
        old_weights = layer["weights"]    # [in_dim][out_dim]
        new_weights = jnp.array(old_weights)          # (in_dim, out_dim)
        old_biases  = layer["biases"]     # [out_dim]
        new_biases  = jnp.array(old_biases)
        params[layer_name] = {'kernel': new_weights, 'bias': new_biases}
    return params

def print_params_info(params):
    """
    Print information about Flax parameters.

    Args:
        params: Flax parameter dictionary
    """
    print("Parameter shapes:")
    total_params = 0

    for layer_name, layer_params in params.items():
        kernel_shape = layer_params['kernel'].shape
        bias_shape = layer_params['bias'].shape
        layer_params_count = kernel_shape[0] * kernel_shape[1] + bias_shape[0]
        total_params += layer_params_count

        print(f"  {layer_name}: weights {kernel_shape}, biases {bias_shape} ({layer_params_count} params)")

    print(f"Total parameters: {total_params}")

if __name__ == "__main__":
    # Test parameter loading/saving
    try:
        params = load_params()
        print("Loaded parameters:")
        print_params_info(params)
    except FileNotFoundError:
        print("No parameters file found. Run initialize_network_jax.py first.")
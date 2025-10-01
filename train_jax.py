import argparse
from pathlib import Path
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training import train_state
import optax

import helpers
import param_io_jax

# ------------------------- JAX Network Definition -------------------------

class MLP(nn.Module):
    features: List[int]

    @nn.compact
    def __call__(self, x):
        # Hidden layers with sigmoid
        for feat in self.features[1:-1]:
            x = nn.Dense(feat)(x)
            x = nn.sigmoid(x)
        # Output layer (logits)
        return nn.Dense(self.features[-1])(x)

def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Cross-entropy loss with softmax"""
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))

@jax.jit
def train_step(state, batch):
    """Single training step"""
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        loss = cross_entropy_loss(logits, batch['label'])
        return loss, logits

    (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def compute_accuracy(state, batch):
    """Compute accuracy for a batch"""
    logits = state.apply_fn({'params': state.params}, batch['image'])
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == batch['label'])

def create_train_state(rng, model, learning_rate):
    """Create initial training state with plain SGD (no momentum)"""
    params = model.init(rng, jnp.ones((1, model.features[0])))['params']
    tx = optax.sgd(learning_rate=learning_rate, momentum=None)  # no momentum
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def load_data(train_dir: str, max_train: int = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Load training data into JAX arrays"""
    root = str(Path(train_dir).parent)
    total_train = helpers.get_split_size("train", root=root)
    n_samples = min(total_train, max_train) if max_train is not None else total_train

    print(f"Loading {n_samples} training images...")

    images, labels = [], []
    for idx in range(n_samples):
        images.append(helpers.get_image_data("train", idx, root=root))
        labels.append(helpers.get_label("train", idx, root=root))
        if (idx + 1) % 10000 == 0 or (idx + 1) == n_samples:
            print(f"  loaded {idx+1}/{n_samples} samples", flush=True)

    images = jnp.array(np.array(images), dtype=jnp.float32)
    labels = jnp.array(np.array(labels), dtype=jnp.int32)
    return images, labels

def train_full_batch(
    train_dir: str,
    net_sizes: List[int],
    epochs: int,
    lr: float,
    seed: int,
    max_train: int = None,
) -> train_state.TrainState:
    """Full-batch training with plain SGD"""
    images, labels = load_data(train_dir, max_train)

    model = MLP(features=net_sizes)
    rng = jax.random.PRNGKey(seed)
    state = create_train_state(rng, model, lr)

    print(f"Training with {images.shape[0]} samples for {epochs} epochs...")

    batch = {'image': images, 'label': labels}
    for epoch in range(1, epochs + 1):
        state, loss = train_step(state, batch)
        accuracy = compute_accuracy(state, batch)
        print(f"Epoch {epoch:3d} | loss {loss:.4f} | acc {accuracy*100:.2f}%")

    return state

def evaluate_accuracy(state, test_dir: str, max_eval: int = 1000) -> float:
    """Evaluate accuracy on test set"""
    root = str(Path(test_dir).parent)
    total = helpers.get_split_size("test", root=root)
    n = min(total, max_eval) if max_eval is not None else total

    print(f"Evaluating on {n} test images...")

    correct = 0
    for idx in range(n):
        arr = helpers.get_image_data("test", idx, root=root)
        image = jnp.array(arr, dtype=jnp.float32).reshape(1, -1)
        label = helpers.get_label("test", idx, root=root)

        logits = state.apply_fn({'params': state.params}, image)
        prediction = jnp.argmax(logits, axis=-1)[0]
        if prediction == label:
            correct += 1

    return correct / max(1, n)

def main():
    ap = argparse.ArgumentParser(description="JAX-based MNIST trainer (SGD only, full-batch)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-root", type=str, default=str(Path("data") / "train"))
    ap.add_argument("--test-root", type=str, default=str(Path("data") / "test"))
    ap.add_argument("--max-train", type=int, default=None, help="limit number of training images")
    ap.add_argument("--max-test", type=int, default=1000, help="limit number of test images for quick eval")
    ap.add_argument("--sizes", type=str, default="784,16,16,10", help="comma-separated layer sizes")

    args = ap.parse_args()
    sizes = [int(s.strip()) for s in args.sizes.split(",")]

    # Train the model
    state = train_full_batch(
        train_dir=args.train_root,
        net_sizes=sizes,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        max_train=args.max_train,
    )

    # Evaluate
    try:
        acc = evaluate_accuracy(state, args.test_root, max_eval=args.max_test)
        print(f"Test accuracy: {acc*100:.2f}%")
    except Exception as e:
        print(f"Eval skipped or failed: {e}")

    # Save parameters
    param_io_jax.save_params(state.params)
    print("Saved trained parameters to params_jax.pkl")

if __name__ == "__main__":
    main()


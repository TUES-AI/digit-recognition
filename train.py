"""
Това е малко-сложна имплементация на тренирането с JAX.

Не е нужно да я четете или разбирате, ще стигнем до такъв код много по-късно в курса.
"""

from __future__ import annotations
import pickle
import struct
import gzip
import time
from pathlib import Path
from typing import List, Dict, Any, Sequence, Tuple, Optional

import numpy as np
import jax
import jax.numpy as jnp
import network as net


def _find_idx_files(split: str, root: str) -> Tuple[Path, Path]:
    split = split.lower()
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split}")

    prefix = "train" if split == "train" else "t10k"
    rootp = Path(root)

    def pick(base: str) -> Path:
        p1 = rootp / base
        p2 = rootp / (base + ".gz")
        if p1.exists():
            return p1
        if p2.exists():
            return p2
        q1 = rootp.parent / base
        q2 = rootp.parent / (base + ".gz")
        if q1.exists():
            return q1
        if q2.exists():
            return q2
        raise FileNotFoundError(
            f"Could not find {base}(.gz) under {rootp} or {rootp.parent}"
        )

    img_path = pick(f"{prefix}-images-idx3-ubyte")
    lbl_path = pick(f"{prefix}-labels-idx1-ubyte")
    return img_path, lbl_path


def _read_idx_images(path: Path) -> np.ndarray:
    open_fn = gzip.open if path.suffix == ".gz" else open
    with open_fn(path, "rb") as f:
        data = f.read()
    magic, num, rows, cols = struct.unpack(">IIII", data[:16])
    if magic != 2051:
        raise ValueError(f"Bad magic for images: {magic} at {path}")
    arr = np.frombuffer(data, dtype=np.uint8, offset=16)
    arr = arr.reshape(num, rows * cols).astype(np.float32) / 255.0
    return arr


def _read_idx_labels(path: Path) -> np.ndarray:
    open_fn = gzip.open if path.suffix == ".gz" else open
    with open_fn(path, "rb") as f:
        data = f.read()
    magic, num = struct.unpack(">II", data[:8])
    if magic != 2049:
        raise ValueError(f"Bad magic for labels: {magic} at {path}")
    arr = np.frombuffer(data, dtype=np.uint8, offset=8)
    if arr.shape[0] != num:
        raise ValueError("Label count mismatch.")
    return arr


def load_split(root: str, split: str, limit: Optional[int] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    img_path, lbl_path = _find_idx_files(split, root)
    images_np = _read_idx_images(img_path)
    labels_np = _read_idx_labels(lbl_path)

    if images_np.shape[0] != labels_np.shape[0]:
        raise ValueError("Images/labels count mismatch.")

    if limit is not None:
        images_np = images_np[:limit]
        labels_np = labels_np[:limit]

    images = jnp.asarray(images_np, dtype=jnp.float32)
    labels = jnp.asarray(labels_np, dtype=jnp.int32)
    return images, labels


def to_jax_params(pickle_params: List[Dict[str, Any]]) -> List[Dict[str, jnp.ndarray]]:
    out: List[Dict[str, jnp.ndarray]] = []
    for layer in pickle_params:
        W = jnp.asarray(layer["weights"], dtype=jnp.float32)
        b = jnp.asarray(layer["biases"], dtype=jnp.float32)
        out.append({"weights": W, "biases": b})
    return out


def to_pickle_params(jax_params: List[Dict[str, jnp.ndarray]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for layer in jax_params:
        W_list = np.array(layer["weights"]).astype(np.float32).tolist()
        b_list = np.array(layer["biases"]).astype(np.float32).tolist()
        out.append({"weights": W_list, "biases": b_list})
    return out


def save_params(jax_params: List[Dict[str, jnp.ndarray]], path: str = "params.pkl") -> None:
    with open(path, "wb") as f:
        pickle.dump(to_pickle_params(jax_params), f)


def load_params(path: str = "params.pkl") -> Optional[List[Dict[str, jnp.ndarray]]]:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        raw = pickle.load(f)
    return to_jax_params(raw)


def glorot_uniform(rng_key: jax.Array, in_dim: int, out_dim: int) -> jax.Array:
    limit = jnp.sqrt(6.0 / (in_dim + out_dim))
    return jax.random.uniform(
        rng_key, (in_dim, out_dim), minval=-limit, maxval=limit, dtype=jnp.float32
    )


def init_params(layer_sizes: Sequence[int], seed: int = 0) -> List[Dict[str, jnp.ndarray]]:
    params: List[Dict[str, jnp.ndarray]] = []
    key = jax.random.PRNGKey(seed)
    for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
        key, kW = jax.random.split(key)
        W = glorot_uniform(kW, int(in_dim), int(out_dim))
        b = jnp.zeros((int(out_dim),), dtype=jnp.float32)
        params.append({"weights": W, "biases": b})
    return params


def forward(params: List[Dict[str, jnp.ndarray]], x: jnp.ndarray) -> jnp.ndarray:
    act = x
    last_idx = len(params) - 1
    for i, layer in enumerate(params):
        W, b = layer["weights"], layer["biases"]
        z = jnp.dot(act, W) + b
        act = jax.nn.sigmoid(z) if i < last_idx else z
    return act


def cross_entropy_loss(params: List[Dict[str, jnp.ndarray]], x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    logits = forward(params, x)
    log_probs = jax.nn.log_softmax(logits)
    nll = -log_probs[jnp.arange(y.shape[0]), y]
    return jnp.mean(nll)


def accuracy(params: List[Dict[str, jnp.ndarray]], x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    preds = jnp.argmax(forward(params, x), axis=-1)
    return jnp.mean((preds == y).astype(jnp.float32))


@jax.jit
def train_step(
    params: List[Dict[str, jnp.ndarray]],
    x: jnp.ndarray,
    y: jnp.ndarray,
    lr: jnp.ndarray,
) -> Tuple[List[Dict[str, jnp.ndarray]], jnp.ndarray]:
    loss, grads = jax.value_and_grad(cross_entropy_loss)(params, x, y)
    new_params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    return new_params, loss


@jax.jit
def eval_metrics(params: List[Dict[str, jnp.ndarray]], x: jnp.ndarray, y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    l = cross_entropy_loss(params, x, y)
    a = accuracy(params, x, y)
    return l, a


def train_full_batch(
    data_root: str = "data",
    param_path: str = "params.pkl",
    epochs: int = 1000,
    lr: float = 0.25,
    max_train: int = 60000,
    seed: int = 0,
    log_every: int = 10,
    test_eval_every: int = 0,
    max_test: Optional[int] = None,
) -> None:
    X_train, y_train = load_split(data_root, "train", limit=max_train)
    X_test,  y_test  = load_split(data_root, "test",  limit=max_test)

    params = load_params(param_path)
    if params is not None:
        in_dim = int(X_train.shape[1])
        out_dim = int(jnp.max(y_train)) + 1
        first_W = params[0]["weights"]
        last_W = params[-1]["weights"]
        if first_W.shape[0] != in_dim:
            raise ValueError(f"Loaded params expect input dim {first_W.shape[0]} but data has {in_dim}.")
        if last_W.shape[1] != out_dim:
            raise ValueError(f"Loaded params expect {last_W.shape[1]} classes but data has {out_dim}.")
    else:
        layer_sizes = [int(X_train.shape[1]), 128, 10]
        params = init_params(layer_sizes, seed=seed)

    lr32 = jnp.float32(lr)

    _ = eval_metrics.lower(params, X_train, y_train).compile()
    _ = eval_metrics.lower(params, X_test,  y_test ).compile()
    _ = train_step.lower(params, X_train, y_train, lr32).compile()

    start_time = time.time()
    try:
        for epoch in range(epochs):
            params, loss = train_step(params, X_train, y_train, lr32)

            if epoch % log_every == 0 or epoch == epochs - 1:
                l_tr, a_tr = eval_metrics(params, X_train, y_train)
                msg = f"Epoch {epoch+1}/{epochs} - train: loss={float(l_tr):.4f}, acc={float(a_tr)*100:.2f}%"

                if test_eval_every > 0 and (epoch % test_eval_every == 0 or epoch == epochs - 1):
                    l_te, a_te = eval_metrics(params, X_test, y_test)
                    msg += f" - test: loss={float(l_te):.4f}, acc={float(a_te)*100:.2f}%"

                print(msg)

        l_te, a_te = eval_metrics(params, X_test, y_test)
        print(f"Final test: loss={float(l_te):.4f}, acc={float(a_te)*100:.2f}%")

        save_params(params, param_path)
        print(f"Saved parameters to {param_path} ")

    except KeyboardInterrupt:
        print("\n[Ctrl+C] Stopping… computing final metrics and saving current params.")
        try:
            l_tr, a_tr = eval_metrics(params, X_train, y_train)
            l_te, a_te = eval_metrics(params, X_test,  y_test)
            print(
                f"Final (interrupted) - train: loss={float(l_tr):.4f}, acc={float(a_tr)*100:.2f}% "
                f"- test: loss={float(l_te):.4f}, acc={float(a_te)*100:.2f}%"
            )
        except Exception as e:
            print(f"(Warning) Could not compute final metrics due to: {e}")

        try:
            save_params(params, param_path)
            print(f"(Interrupted) Saved parameters to {param_path}")
        except Exception as e:
            print(f"(Warning) Could not save parameters due to: {e}")


@jax.jit
def predict_logits(params: List[Dict[str, jnp.ndarray]], x: jnp.ndarray) -> jnp.ndarray:
    x2 = x if x.ndim == 2 else x[None, ...]
    return forward(params, x2)


def predict(params: List[Dict[str, jnp.ndarray]], x: jnp.ndarray) -> np.ndarray:
    logits = predict_logits(params, x)
    return np.asarray(jnp.argmax(logits, axis=-1))


if __name__ == "__main__":
    train_full_batch(
        data_root="data",
        param_path="params.pkl",
        epochs=net.number_of_epochs,
        lr=net.learning_rate,
        max_train=net.images_to_train_on,
        seed=np.random.randint(0, 2**31 - 1),
        log_every=10,
        test_eval_every=0,
        max_test=None,
    )


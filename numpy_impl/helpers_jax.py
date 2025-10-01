# helpers_jax.py
# JAX-compatible version of helpers.py

import math
import struct
import gzip
from pathlib import Path
from typing import Tuple
import numpy as np
import jax.numpy as jnp

# Cache for quick random access
# _CACHE[split] = (images_np_float32[N, 784], labels_np_uint8[N])
_CACHE = {}

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
        # Also allow binaries under root's parent if training uses data/train, data/test
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
    # >IIII: magic(2051), num_images, rows, cols
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
    # >II: magic(2049), num_labels
    magic, num = struct.unpack(">II", data[:8])
    if magic != 2049:
        raise ValueError(f"Bad magic for labels: {magic} at {path}")
    arr = np.frombuffer(data, dtype=np.uint8, offset=8)
    if arr.shape[0] != num:
        raise ValueError("Label count mismatch.")
    return arr

def _ensure_loaded(split: str, root: str = "data") -> None:
    s = split.lower()
    if s in _CACHE:
        return
    img_path, lbl_path = _find_idx_files(s, root)
    images = _read_idx_images(img_path)
    labels = _read_idx_labels(lbl_path)
    if images.shape[0] != labels.shape[0]:
        raise ValueError("Images/labels count mismatch.")
    _CACHE[s] = (images, labels)

def get_image_data(split: str, index: int, root: str = "data") -> np.ndarray:
    """
    Returns a 784-length float32 numpy array in [0,1] for the given image.
    """
    _ensure_loaded(split, root)
    return _CACHE[split.lower()][0][index]

def get_image_data_jax(split: str, index: int, root: str = "data") -> jnp.ndarray:
    """
    Returns a 784-length JAX array in [0,1] for the given image.
    """
    _ensure_loaded(split, root)
    return jnp.array(_CACHE[split.lower()][0][index], dtype=jnp.float32)

def get_label(split: str, index: int, root: str = "data") -> int:
    """
    Returns the integer label for the given image.
    """
    _ensure_loaded(split, root)
    return int(_CACHE[split.lower()][1][index])

def get_label_jax(split: str, index: int, root: str = "data") -> jnp.ndarray:
    """
    Returns the integer label as JAX array for the given image.
    """
    _ensure_loaded(split, root)
    return jnp.array(_CACHE[split.lower()][1][index], dtype=jnp.int32)

def get_split_size(split: str, root: str = "data") -> int:
    """
    Returns number of samples in the split.
    """
    _ensure_loaded(split, root)
    return int(_CACHE[split.lower()][0].shape[0])

def print_ascii(img_array) -> None:
    """
    Print ASCII art representation of an image array.
    Works with both numpy and JAX arrays.
    """
    # Convert to numpy if it's a JAX array
    if hasattr(img_array, '__jax_array__'):
        img_array = np.array(img_array)

    for i in range(int(math.sqrt(len(img_array)))):
        for j in range(int(math.sqrt(len(img_array)))):
            value = img_array[int(math.sqrt(len(img_array)))*i + j]
            if  value > 0.5:
                print("#", end="")
            elif value > 0.25:
                print(".", end="")
            else:
                print(" ", end="")
        print()

def get_batch_data(split: str, indices: list, root: str = "data") -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Get batch of images and labels as JAX arrays.
    """
    _ensure_loaded(split, root)
    cache = _CACHE[split.lower()]

    images = np.stack([cache[0][idx] for idx in indices])
    labels = np.array([cache[1][idx] for idx in indices])

    return jnp.array(images, dtype=jnp.float32), jnp.array(labels, dtype=jnp.int32)
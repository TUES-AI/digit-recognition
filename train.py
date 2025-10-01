"""
Pure-Python full-batch gradient descent trainer for MNIST PNGs.

- No numpy / no PyTorch in this file.
- Uses helpers.py only to read per-image flattened pixel data (784 floats in [0,1]).
- Uses param_io.py to save the trained network exactly as a list of layers:
    layer = {"weights": [[...]*out_dim for in_dim], "biases": [0.0]*out_dim}
- Weight shape is [in_dim][out_dim], same as initialize_network.py.
- Activation: sigmoid on hidden layers; softmax + cross-entropy at output.
- Optimization: full-batch gradient descent (one gradient over the whole train set per epoch).
- CLI:
    python train.py --epochs 5 --lr 0.1 --train-root data/train --seed 42
    Optional: --max-train N   (limit number of training images, for quick tests)

This is intentionally minimal to match your "bare minimum" plan.
"""

import argparse
import glob
import math
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import helpers             # ok to use; returns numpy arrays but we convert to Python lists
import param_io            # save_params(network)

# ------------------------- Utilities -------------------------

def parse_label_from_filename(path: str) -> int:
    """Expect paths like data/train/00000012-7.png -> return 7"""
    name = os.path.basename(path)
    # split on '-', then take the numeric part before .png
    try:
        after_dash = name.split('-', 1)[1]
        label_str = after_dash.split('.', 1)[0]
        return int(label_str)
    except Exception:
        raise ValueError(f"Cannot parse label from filename: {name}")

def list_indexed_files(root_dir: str) -> List[str]:
    """
    Return a list of files sorted by their numeric index prefix (00000000-*.png, 00000001-*.png, ...).
    """
    pattern = str(Path(root_dir) / "*-*.png")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No PNGs found under {root_dir}. Did you run the downloader?")
    # sort by the integer value of the 8-digit prefix
    def idx_key(p: str) -> int:
        base = os.path.basename(p)
        # first 8 chars should be zero-padded index
        try:
            return int(base.split('-', 1)[0])
        except Exception:
            return sys.maxsize
    files.sort(key=idx_key)
    return files

def one_hot(label: int, num_classes: int) -> List[float]:
    v = [0.0] * num_classes
    v[label] = 1.0
    return v

def sigmoid(x: float) -> float:
    # clip for numerical safety (not strictly necessary on MNIST)
    if x < -60.0: 
        return 0.0
    if x >  60.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))

def dsigmoid_from_output(s: float) -> float:
    # derivative using already-computed sigmoid output s
    return s * (1.0 - s)

def softmax(logits: List[float]) -> List[float]:
    # stable softmax
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    s = sum(exps)
    return [e / s for e in exps]

def dot(u: List[float], v: List[float]) -> float:
    return sum(uu * vv for uu, vv in zip(u, v))

# ------------------------- Network (list-of-layers) -------------------------
# Each layer: {"weights": [[in_dim x out_dim]], "biases": [out_dim]}
# Hidden activations: sigmoid; Output: logits (then softmax in loss)

def init_layer(in_dim: int, out_dim: int, rng: random.Random) -> Dict[str, List]:
    # Xavier/Glorot uniform for sigmoid
    a = math.sqrt(6.0 / (in_dim + out_dim))
    weights = []
    for _ in range(in_dim):
        row = [(rng.random() * 2.0 - 1.0) * a for __ in range(out_dim)]
        weights.append(row)
    biases = [0.0 for _ in range(out_dim)]
    return {"weights": weights, "biases": biases}

def init_network(sizes: List[int], seed: int) -> List[Dict[str, List]]:
    rng = random.Random(seed)
    net = []
    for i in range(1, len(sizes)):
        net.append(init_layer(sizes[i-1], sizes[i], rng))
    return net

def forward_collect(net: List[Dict[str, List]], x: List[float]) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Forward pass collecting per-layer pre-activations (z) and activations (a).
    Return (zs, activations).
    activations[0] is the input x.
    For last layer, activations[L] is *logits* (no softmax here).
    """
    a = x
    activations = [a]         # a^0 = x
    zs = []                   # z^1 ... z^L
    for li, layer in enumerate(net):
        W = layer["weights"]  # [in][out]
        b = layer["biases"]   # [out]
        out_dim = len(b)
        # compute z_j = b_j + sum_i a_i * W[i][j]
        z = [0.0] * out_dim
        for j in range(out_dim):
            s = b[j]
            # accumulate dot(a, column j of W)
            for i in range(len(a)):
                s += a[i] * W[i][j]
            z[j] = s
        zs.append(z)
        # activation
        if li < len(net) - 1:
            a = [sigmoid(zj) for zj in z]
        else:
            # last layer: keep logits (no activation here); softmax happens in loss/grad
            a = z[:]  # logits
        activations.append(a)
    return zs, activations

# ------------------------- Full-batch loss & grad -------------------------

def zero_like_network(net: List[Dict[str, List]]) -> List[Dict[str, List]]:
    grads = []
    for layer in net:
        in_dim = len(layer["weights"])
        out_dim = len(layer["weights"][0])
        gW = [ [0.0]*out_dim for _ in range(in_dim) ]
        gb = [ 0.0 for _ in range(out_dim) ]
        grads.append({"weights": gW, "biases": gb})
    return grads

def add_grads_(acc: List[Dict[str, List]], g: List[Dict[str, List]]) -> None:
    for accL, gL in zip(acc, g):
        # weights
        for i in range(len(accL["weights"])):
            ai = accL["weights"][i]
            gi = gL["weights"][i]
            for j in range(len(ai)):
                ai[j] += gi[j]
        # biases
        for j in range(len(accL["biases"])):
            accL["biases"][j] += gL["biases"][j]

def scale_grads_(g: List[Dict[str, List]], scale: float) -> None:
    for layer in g:
        for i in range(len(layer["weights"])):
            row = layer["weights"][i]
            for j in range(len(row)):
                row[j] *= scale
        for j in range(len(layer["biases"])):
            layer["biases"][j] *= scale

# --------- NEW: optimizer state & updates (SGD / Adam / AdamW) ---------

def init_opt_state(net: List[Dict[str, List]], opt: str):
    """
    Create optimizer state with same shapes as net's weights/biases.
    For SGD: empty state (None).
    For Adam/AdamW: m and v for weights & biases.
    """
    if opt == "sgd":
        return None
    state = []
    for layer in net:
        in_dim = len(layer["weights"])
        out_dim = len(layer["weights"][0])
        # zeros with the same shapes
        mW = [ [0.0]*out_dim for _ in range(in_dim) ]
        vW = [ [0.0]*out_dim for _ in range(in_dim) ]
        mB = [0.0 for _ in range(out_dim)]
        vB = [0.0 for _ in range(out_dim)]
        state.append({"mW": mW, "vW": vW, "mB": mB, "vB": vB})
    return state

def apply_update_(
    net: List[Dict[str, List]],
    g: List[Dict[str, List]],
    lr: float,
    opt: str,
    opt_state,
    t: int,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
) -> None:
    """
    In-place parameter update.
    - SGD:    W -= lr * gW;  b -= lr * gb
    - Adam:   Adam with bias correction
    - AdamW:  Adam + decoupled weight decay (weights only)
    """
    if opt == "sgd" or opt_state is None:
        # vanilla SGD
        for layer, grad in zip(net, g):
            W = layer["weights"]; b = layer["biases"]
            gW = grad["weights"]; gb = grad["biases"]
            for i in range(len(W)):
                Wi = W[i]; gWi = gW[i]
                for j in range(len(Wi)):
                    Wi[j] -= lr * gWi[j]
            for j in range(len(b)):
                b[j] -= lr * gb[j]
        return

    # Adam / AdamW
    b1t = 1.0 - (beta1 ** t)
    b2t = 1.0 - (beta2 ** t)

    for L, (layer, grad, st) in enumerate(zip(net, g, opt_state)):
        W = layer["weights"]; b = layer["biases"]
        gW = grad["weights"]; gb = grad["biases"]
        mW = st["mW"]; vW = st["vW"]
        mB = st["mB"]; vB = st["vB"]

        # weights
        for i in range(len(W)):
            Wi = W[i]; gWi = gW[i]; mWi = mW[i]; vWi = vW[i]
            for j in range(len(Wi)):
                g_ij = gWi[j]
                # Adam moments
                m_ij = mWi[j] = beta1 * mWi[j] + (1.0 - beta1) * g_ij
                v_ij = vWi[j] = beta2 * vWi[j] + (1.0 - beta2) * (g_ij * g_ij)
                # bias-corrected
                m_hat = m_ij / (b1t if b1t != 0.0 else 1.0)
                v_hat = v_ij / (b2t if b2t != 0.0 else 1.0)
                step = m_hat / (math.sqrt(v_hat) + eps)
                # parameter update
                Wi[j] -= lr * step
                if opt == "adamw" and weight_decay != 0.0:
                    # decoupled weight decay on weights only
                    Wi[j] -= lr * weight_decay * Wi[j]

        # biases (no weight decay)
        for j in range(len(b)):
            g_j = gb[j]
            m_j = mB[j] = beta1 * mB[j] + (1.0 - beta1) * g_j
            v_j = vB[j] = beta2 * vB[j] + (1.0 - beta2) * (g_j * g_j)
            m_hat = m_j / (b1t if b1t != 0.0 else 1.0)
            v_hat = v_j / (b2t if b2t != 0.0 else 1.0)
            step = m_hat / (math.sqrt(v_hat) + eps)
            b[j] -= lr * step

# -----------------------------------------------------------------------

def ce_loss_and_grad_single(net: List[Dict[str, List]], x: List[float], label: int) -> Tuple[float, List[Dict[str, List]]]:
    """
    Cross-entropy with softmax at output. Returns (loss, grads for all layers).
    Layer indices:
      l = 0 .. L-1 are weight layers
      a^0 = input; for l>=1: a^l are hidden activations (sigmoid); a^L are logits
      z^l are pre-activations for layer l (so z^1 is first hidden pre-activation)
    Deltas:
      delta^{L} = softmax(logits) - one_hot
      For l = L-1 .. 1: delta^{l} = (W^{l+1} * delta^{l+1}) ⊙ sigmoid'(z^{l})
    Gradients:
      dW^{l}[i][j] = a^{l}[i] * delta^{l+1}[j]
      db^{l}[j]    = delta^{l+1}[j]
    """
    zs, activations = forward_collect(net, x)  # zs[0]=z^1, activations[0]=a^0, activations[-1]=a^L (logits)
    logits = activations[-1]
    probs = softmax(logits)
    loss = -math.log(max(1e-12, probs[label]))

    L = len(net)  # number of weight layers
    # Initialize deltas for layers 1..L (we'll keep index aligned so delta[0] unused)
    deltas: List[List[float]] = [None] * (L + 1)

    # Output delta (size = size of layer L outputs)
    delta_L = [p for p in probs]
    delta_L[label] -= 1.0
    deltas[L] = delta_L

    # Backprop deltas through hidden layers: l = L-1 down to 1
    for l in range(L-1, 0, -1):
        Wl_plus_1 = net[l]["weights"]      # W^{l+1} has shape [in_l][out_l]
        in_dim = len(Wl_plus_1)            # = size(a^{l})
        out_dim = len(Wl_plus_1[0])        # = size(a^{l+1})
        delta_next = deltas[l+1]           # size out_dim
        # delta^{l}[i] = sigmoid'(z^{l}[i]) * sum_j W^{l+1}[i][j] * delta^{l+1}[j]
        dl = [0.0] * in_dim
        a_l = activations[l]               # = sigmoid(z^{l})
        for i in range(in_dim):
            s = 0.0
            row = Wl_plus_1[i]
            for j in range(out_dim):
                s += row[j] * delta_next[j]
            s *= dsigmoid_from_output(a_l[i])
            dl[i] = s
        deltas[l] = dl

    # Now compute gradients for each layer l = 0..L-1 using a^{l} and delta^{l+1}
    grads = zero_like_network(net)
    for l in range(L):
        a_l = activations[l]        # size in_l
        delta_next = deltas[l+1]    # size out_l
        gW = grads[l]["weights"]
        gb = grads[l]["biases"]
        in_dim = len(a_l)
        out_dim = len(delta_next)
        for i in range(in_dim):
            ai = a_l[i]
            row = gW[i]
            for j in range(out_dim):
                row[j] += ai * delta_next[j]
        for j in range(out_dim):
            gb[j] += delta_next[j]

    return loss, grads

def train_full_batch(
    train_dir: str,
    net_sizes: List[int],
    epochs: int,
    lr: float,
    seed: int,
    max_train: int = None,
    opt: str = "sgd",
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> List[Dict[str, List]]:
    # Initialize (or re-init) network
    net = init_network(net_sizes, seed=seed)

    # Use MNIST IDX binaries via helpers (root is parent of split folder)
    root = str(Path(train_dir).parent)
    total_train = helpers.get_split_size("train", root=root)
    n_samples = min(total_train, max_train) if max_train is not None else total_train
    print(f"Found {n_samples} training images (from IDX binaries) under {root}")

    # -------- preload all training images/labels into memory once --------
    data_x: List[List[float]] = []
    data_y: List[int] = []
    for idx in range(n_samples):
        arr = helpers.get_image_data("train", idx, root=root)
        x = [float(v) for v in list(arr)]   # keep 'no numpy in this file'
        y = helpers.get_label("train", idx, root=root)
        data_x.append(x)
        data_y.append(y)
        if (idx + 1) % 10000 == 0 or (idx + 1) == n_samples:
            print(f"  loaded {idx+1}/{n_samples} samples into memory", flush=True)
    # --------------------------------------------------------------------

    # Optimizer state
    opt_state = init_opt_state(net, opt)

    for epoch in range(1, epochs+1):
        total_loss = 0.0
        full_grad = zero_like_network(net)

        for idx in range(n_samples):
            x = data_x[idx]
            y = data_y[idx]
            loss, grads = ce_loss_and_grad_single(net, x, y)
            total_loss += loss
            add_grads_(full_grad, grads)

        # Average the loss and grads (full-batch)
        avg_loss = total_loss / n_samples
        scale_grads_(full_grad, 1.0 / n_samples)

        # Update (t = epoch for bias correction since we have 1 update per epoch)
        apply_update_(
            net, full_grad, lr,
            opt=opt, opt_state=opt_state, t=epoch,
            beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay,
        )

        print(f"Epoch {epoch:3d} | loss {avg_loss:.4f}")

    return net

def evaluate_accuracy(net: List[Dict[str, List]], eval_dir: str, max_eval: int = 1000) -> float:
    root = str(Path(eval_dir).parent)
    total = helpers.get_split_size("test", root=root)
    n = min(total, max_eval) if max_eval is not None else total

    correct = 0
    for idx in range(n):
        arr = helpers.get_image_data("test", idx, root=root)
        x = [float(v) for v in list(arr)]
        _, activations = forward_collect(net, x)
        probs = softmax(activations[-1])
        pred = max(range(len(probs)), key=lambda k: probs[k])
        y = helpers.get_label("test", idx, root=root)
        if pred == y:
            correct += 1
    return correct / max(1, n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-root", type=str, default=str(Path("data") / "train"))
    ap.add_argument("--test-root", type=str, default=str(Path("data") / "test"))
    ap.add_argument("--max-train", type=int, default=None, help="limit number of training images")
    ap.add_argument("--max-test", type=int, default=1000, help="limit number of test images for quick eval")
    # network sizes fixed to match initialize_network.py for compatibility
    ap.add_argument("--sizes", type=str, default="784,16,16,10", help="comma-separated layer sizes")

    # NEW: optimizer options
    ap.add_argument("--opt", type=str, default="sgd", choices=["sgd", "adam", "adamw"], help="optimizer")
    ap.add_argument("--beta1", type=float, default=0.9, help="Adam/AdamW beta1")
    ap.add_argument("--beta2", type=float, default=0.999, help="Adam/AdamW beta2")
    ap.add_argument("--eps", type=float, default=1e-8, help="Adam/AdamW epsilon")
    ap.add_argument("--weight-decay", type=float, default=0.0, help="AdamW decoupled weight decay (weights only)")

    args = ap.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",")]
    net = train_full_batch(
        train_dir=args.train_root,
        net_sizes=sizes,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        max_train=args.max_train,
        opt=args.opt,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # (Optional) quick eval on a subset of test to confirm it learns something
    try:
        acc = evaluate_accuracy(net, args.test_root, max_eval=args.max_test)
        print(f"Quick eval accuracy on first {args.max_test} test images: {acc*100:.2f}%")
    except Exception as e:
        print(f"Eval skipped or failed: {e}")

    # Save params in the expected format
    param_io.save_params(net)
    print("Saved trained parameters to params.pkl")

if __name__ == "__main__":
    main()


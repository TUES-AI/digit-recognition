import argparse
import gzip
import os
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import Tuple
import helpers

try:
    from PIL import Image
except Exception as e:
    print("This script requires the 'Pillow' package. Install with: pip install Pillow", file=sys.stderr)
    raise

import urllib.request
import urllib.error

def _human(n: float) -> str:
    for unit in ["B","KB","MB","GB"]:
        if n < 1024.0:
            return f"{n:3.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"

def stream_download(url: str, dest: Path, timeout: int = 30) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as fh:
        total = resp.headers.get("Content-Length")
        total = int(total) if total is not None else None
        downloaded = 0
        chunk = 1024 * 64
        start = time.time()
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            fh.write(buf)
            downloaded += len(buf)
            elapsed = time.time() - start
            if elapsed >= 0.5:
                start = time.time()
                if total:
                    pct = 100.0 * downloaded / total
                    print(f"\r→ {dest.name}: {_human(downloaded)} / {_human(total)} ({pct:4.1f}%)", end="", flush=True)
                else:
                    print(f"\r→ {dest.name}: {_human(downloaded)}", end="", flush=True)
        print(f"\r✓ Downloaded {dest.name} ({_human(downloaded)})")
    tmp.replace(dest)

def load_idx_images(path: Path) -> "tuple[int, int, int, bytes]":
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 0x00000803:
            raise ValueError(f"{path.name}: unexpected magic {magic}")
        data = f.read()
    expected = num * rows * cols
    if len(data) != expected:
        raise ValueError(f"{path.name}: size mismatch, expected {expected}, got {len(data)}")
    return num, rows, cols, data

def load_idx_labels(path: Path) -> "tuple[int, bytes]":
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 0x00000801:
            raise ValueError(f"{path.name}: unexpected magic {magic}")
        data = f.read()
    if len(data) != num:
        raise ValueError(f"{path.name}: size mismatch, expected {num}, got {len(data)}")
    return num, data

def save_pngs(images: Tuple[int, int, int, bytes],
              labels: Tuple[int, bytes],
              out_dir: Path,
              start_index: int = 0) -> None:
    n_img, rows, cols, img_bytes = images
    n_lbl, lbl_bytes = labels
    assert n_img == n_lbl, f"image/label count mismatch: {n_img} vs {n_lbl}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_img):
        label = lbl_bytes[i]
        offset = i * rows * cols
        buf = img_bytes[offset: offset + rows * cols]
        im = Image.frombytes("L", (cols, rows), buf)
        name = f"{i + start_index:08d}-{label}.png"
        im.save(out_dir / name, format="PNG")
        if (i + 1) % 5000 == 0 or i in (0, 1, 2, 3, 4, 9):
            print(f"  wrote {name}")

def main():
    parser = argparse.ArgumentParser(description="Download MNIST and export to PNGs")
    parser.add_argument("--workdir", default="mnist", help="temporary work directory (default: ./mnist)")
    parser.add_argument("--outdir", default="data", help="final output directory (default: ./data)")
    args = parser.parse_args()

    base = Path(os.getcwd())
    work = base / args.workdir
    work.mkdir(parents=True, exist_ok=True)

    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images":  "t10k-images-idx3-ubyte.gz",
        "test_labels":  "t10k-labels-idx1-ubyte.gz",
    }

    label_base = "https://storage.googleapis.com/cvdf-datasets/mnist"
    urls = {
        files["train_images"]: f"{label_base}/train-images-idx3-ubyte.gz",
        files["train_labels"]: f"{label_base}/train-labels-idx1-ubyte.gz",
        files["test_images"]:  f"{label_base}/t10k-images-idx3-ubyte.gz",
        files["test_labels"]:  f"{label_base}/t10k-labels-idx1-ubyte.gz",
    }

    for fname, url in urls.items():
        dest = work / fname
        if dest.exists():
            print(f"Skipping existing {dest.name}")
        else:
            stream_download(url, dest)

    print("Reading IDX data...")
    train_images = load_idx_images(work / files["train_images"])
    test_images  = load_idx_images(work / files["test_images"])
    train_labels = load_idx_labels(work / files["train_labels"])
    test_labels  = load_idx_labels(work / files["test_labels"])

    assert train_images[0] == 60000 and test_images[0] == 10000, "Unexpected MNIST counts"
    assert train_labels[0] == 60000 and test_labels[0] == 10000, "Unexpected MNIST label counts"

    out_base = base / args.outdir
    out_train = out_base / "train"
    out_test  = out_base / "test"

    print(f"Saving train PNGs to {out_train}...")
    save_pngs(train_images, train_labels, out_train, start_index=0)

    print(f"Saving test PNGs to {out_test}...")
    save_pngs(test_images, test_labels, out_test, start_index=0)

    print("Cleaning up...")
    try:
        shutil.rmtree(work)
    except Exception as e:
        print(f"Warning: could not remove work dir {work}: {e}", file=sys.stderr)

    print("\nDone. Remaining items in directory:")
    for p in sorted(base.iterdir()):
        if p.is_dir():
            print(f"  [dir] {p.name}")
        else:
            print(f"  {p.name}")

if __name__ == "__main__":
    main()
    helpers.print_ascii(helpers.get_image_data("test", 0))
    print("This should be a 7 or another number if eveything is setup correctly.")

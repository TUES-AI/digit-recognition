# download.py
import argparse
import io
import os
import gzip
import shutil
from pathlib import Path
from urllib.request import urlretrieve

# MNIST sources (Google Cloud Storage mirror)
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
URLS = {
    "train-images-idx3-ubyte.gz": f"{BASE_URL}/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": f"{BASE_URL}/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz":  f"{BASE_URL}/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz":  f"{BASE_URL}/t10k-labels-idx1-ubyte.gz",
}

def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip] {dest.name} already exists")
        return
    print(f"[down] {dest.name}")
    urlretrieve(url, dest.as_posix())

def gunzip_to_ubyte(gz_path: Path, out_path: Path) -> None:
    if out_path.exists():
        print(f"[skip] {out_path.name} already exists")
        return
    print(f"[unzip] {gz_path.name} -> {out_path.name}")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data", help="output directory for MNIST binaries")
    # Kept for compatibility with projects that also want PNGs; defaults False to avoid extra work
    ap.add_argument("--export-png", action="store_true", help="also export PNGs under data/train and data/test")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Download the 4 .gz binaries (kept on disk — no deletion)
    for fname, url in URLS.items():
        download_file(url, out / fname)

    # 2) Unzip to .ubyte for faster local access (also kept on disk)
    for fname in URLS.keys():
        gz_path = out / fname
        raw_name = fname[:-3] if fname.endswith(".gz") else fname
        raw_path = out / raw_name
        gunzip_to_ubyte(gz_path, raw_path)

    # 3) Optional: export PNGs (disabled by default)
    if args.export_png:
        try:
            from PIL import Image
            import numpy as np
            import struct

            def read_images(raw_path: Path):
                with open(raw_path, "rb") as f:
                    data = f.read()
                magic, num, rows, cols = struct.unpack(">IIII", data[:16])
                assert magic == 2051
                arr = np.frombuffer(data, dtype=np.uint8, offset=16)
                return arr.reshape(num, rows, cols)

            def read_labels(raw_path: Path):
                with open(raw_path, "rb") as f:
                    data = f.read()
                magic, num = struct.unpack(">II", data[:8])
                assert magic == 2049
                arr = np.frombuffer(data, dtype=np.uint8, offset=8)
                return arr

            def export(split: str, img_raw: str, lbl_raw: str):
                imgs = read_images(out / img_raw)
                lbls = read_labels(out / lbl_raw)
                d = out / split
                d.mkdir(parents=True, exist_ok=True)
                for i in range(imgs.shape[0]):
                    im = Image.fromarray(imgs[i], mode="L")
                    label = int(lbls[i])
                    im.save(d / f"{i:08d}-{label}.png")

            export("train", "train-images-idx3-ubyte", "train-labels-idx1-ubyte")
            export("test",  "t10k-images-idx3-ubyte",  "t10k-labels-idx1-ubyte")
            print("[ok] PNG export complete.")
        except Exception as e:
            print(f"[warn] PNG export failed (Pillow/numpy missing?): {e}")

    print("[ok] MNIST binaries ready (gz + ubyte kept).")

if __name__ == "__main__":
    main()


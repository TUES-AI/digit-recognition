import math
from pathlib import Path
import glob
from PIL import Image
import numpy as np

def get_image_data(split: str, index: int, root: str = "data") -> np.ndarray:
    split = split.lower()
    pattern = str(Path(root) / split / f"{index:08d}-*.png")
    files = sorted(glob.glob(pattern))
    path = files[0]
    img = Image.open(path).convert("L")
    if img.size != (28, 28):
        img = img.resize((28, 28), Image.NEAREST)
    arr = np.asarray(img) / 255.0
    return arr.ravel(), int(path.split("-")[-1].split(".")[0])

def print_ascii(img_array: np.ndarray) -> None:
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

"""
Parses the CICY raw text file into:
- X : padded configuration matrices (N, 12, 15)
- y : Hodge numbers (N, 2) → [h11, h21]
"""

import numpy as np
import re
from pathlib import Path

RAW_FILE = "data/raw/cicy3folds.txt"
OUT_X = "data/processed/X.npy"
OUT_Y = "data/processed/y_hodge.npy"

MAX_PS = 12
MAX_POL = 15


def parse_int_list(line):
    """Extract integers from a line like {1, 2, 3}"""
    return list(map(int, re.findall(r"-?\d+", line)))


def main():
    with open(RAW_FILE, "r") as f:
        lines = f.readlines()

    X = []
    y = []

    i = 0
    n = len(lines)

    while i < n:
        line = lines[i].strip()

        # Start of a new manifold block
        if line.startswith("Num"):
            # --- Read header ---
            numps = int(lines[i + 1].split(":")[1])
            numpol = int(lines[i + 2].split(":")[1])
            h11 = int(lines[i + 4].split(":")[1])
            h21 = int(lines[i + 5].split(":")[1])

            # --- Skip to matrix ---
            matrix_start = i + 8
            matrix = []

            for r in range(numps):
                row = parse_int_list(lines[matrix_start + r])
                matrix.append(row)

            matrix = np.array(matrix, dtype=np.float32)

            # --- Pad to (12, 15) ---
            padded = np.zeros((MAX_PS, MAX_POL), dtype=np.float32)
            padded[:matrix.shape[0], :matrix.shape[1]] = matrix

            X.append(padded)
            y.append([h11, h21])

            # Move index forward
            i = matrix_start + numps
        else:
            i += 1

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print(f"Processed {len(X)} CICY 3-folds")
    print(f"X shape      : {X.shape}")
    print(f"Hodge shape  : {y.shape}")
    print(f"Unique h11   : {len(np.unique(y[:,0]))}")
    print(f"Unique h21   : {len(np.unique(y[:,1]))}")

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    np.save(OUT_X, X)
    np.save(OUT_Y, y)


if __name__ == "__main__":
    main()

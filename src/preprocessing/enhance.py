"""
Enhance the dataset by adding scalar geometric features
derived from the CICY configuration matrix.

Currently added feature:
- Ambient factor count (number of non-zero rows)
"""

import numpy as np
import os


def compute_ambient_factor_count(X):
    """
    Number of projective space factors in the ambient space.
    This equals the number of non-zero rows in the CICY matrix.
    """
    # Row is non-zero if any entry in that row is non-zero
    return np.sum(np.any(X != 0, axis=2), axis=1)


if __name__ == "__main__":
    in_dir = "data/processed"
    out_dir = "data/processed"

    X = np.load(os.path.join(in_dir, "X_cicy3.npy")).astype(np.float32)
    y = np.load(os.path.join(in_dir, "y_hodge.npy"))

    # -----------------------------
    # Compute scalar feature(s)
    # -----------------------------
    ambient_factors = compute_ambient_factor_count(X).reshape(-1, 1)

    # -----------------------------
    # Flatten matrix + append scalar
    # -----------------------------
    X_flat = X.reshape(len(X), -1)
    X_enhanced = np.hstack([X_flat, ambient_factors]).astype(np.float32)

    np.save(os.path.join(out_dir, "X_enhanced.npy"), X_enhanced)

    print("Enhanced dataset created")
    print(f"X_enhanced shape: {X_enhanced.shape}")
    print("Added scalar features: [ambient_factor_count]")

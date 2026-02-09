"""
Performs error analysis on the trained CICY CNN model to identify
manifolds most misclassified in terms of h^{2,1}.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from src.models.cnn_model import CICYClassifier


# -----------------------------
# Load data
# -----------------------------
X_enhanced = np.load("data/processed/X_enhanced.npy").astype(np.float32)
y_actual = np.load("data/processed/y_hodge.npy").astype(np.int64)

X_img = X_enhanced[:, :180].reshape(-1, 1, 12, 15)
X_scalar = X_enhanced[:, 180:]


# -----------------------------
# Load model
# -----------------------------
model = CICYClassifier()
model.load_state_dict(
    torch.load("models/cicy_cnn_v1.pt", map_location="cpu")
)
model.eval()


# -----------------------------
# Inference
# -----------------------------
with torch.no_grad():
    img_tensor = torch.from_numpy(X_img)
    scalar_tensor = torch.from_numpy(X_scalar)

    _, out_h21 = model(img_tensor, scalar_tensor)
    pred_h21 = torch.argmax(out_h21, dim=1).numpy()


# -----------------------------
# Error analysis (classification-aware)
# -----------------------------
true_h21 = y_actual[:, 1]
errors = np.abs(pred_h21 - true_h21)

worst_indices = np.argsort(errors)[-5:][::-1]


print("\n--- ERROR ANALYSIS: WORST OFFENDERS (h^{2,1}) ---")

for idx in worst_indices:
    print(f"\nManifold index: {idx}")
    print(f"Actual h^{{2,1}}: {true_h21[idx]}")
    print(f"Predicted h^{{2,1}}: {pred_h21[idx]}")
    print(f"|Δh^{{2,1}}| = {errors[idx]}")

    # Recover matrix image
    matrix = X_enhanced[idx, :180].reshape(12, 15)

    # Print only non-zero submatrix
    nz_rows = ~np.all(matrix == 0, axis=1)
    nz_cols = ~np.all(matrix == 0, axis=0)

    print("Configuration matrix (non-zero block):")
    print(matrix[nz_rows][:, nz_cols])

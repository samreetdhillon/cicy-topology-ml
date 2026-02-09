"""
Evaluation script for the CICY CNN model.
Loads the trained model, runs inference,
and plots predicted vs actual Hodge numbers.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import matplotlib.pyplot as plt
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
model.load_state_dict(torch.load("models/cicy_cnn_v1.pt", map_location="cpu"))
model.eval()


# -----------------------------
# Inference
# -----------------------------
with torch.no_grad():
    img_tensor = torch.from_numpy(X_img)
    scalar_tensor = torch.from_numpy(X_scalar)

    out_h11, out_h21 = model(img_tensor, scalar_tensor)

    pred_h11 = torch.argmax(out_h11, dim=1).numpy()
    pred_h21 = torch.argmax(out_h21, dim=1).numpy()


# -----------------------------
# Plot results
# -----------------------------
os.makedirs("plots", exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# h^{1,1}
ax1.scatter(y_actual[:, 0], pred_h11, alpha=0.3)
ax1.plot(
    [y_actual[:, 0].min(), y_actual[:, 0].max()],
    [y_actual[:, 0].min(), y_actual[:, 0].max()],
    "r--"
)
ax1.set_title(r"$h^{1,1}$ Prediction")
ax1.set_xlabel("Actual")
ax1.set_ylabel("Predicted")

# h^{2,1}
ax2.scatter(y_actual[:, 1], pred_h21, alpha=0.3)
ax2.plot(
    [y_actual[:, 1].min(), y_actual[:, 1].max()],
    [y_actual[:, 1].min(), y_actual[:, 1].max()],
    "r--"
)
ax2.set_title(r"$h^{2,1}$ Prediction")
ax2.set_xlabel("Actual")
ax2.set_ylabel("Predicted")

plt.tight_layout()
plt.savefig("plots/results_v1.png", dpi=150)
plt.show()

print("Evaluation complete. Plot saved to plots/results_v1.png")

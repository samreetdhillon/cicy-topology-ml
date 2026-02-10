"""
Evaluates exact classification accuracy of the trained CICY CNN model.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from src.models.cnn_model import CICYClassifier

# Load data
X_enhanced = np.load("data/processed/X_enhanced.npy").astype(np.float32)
y_actual = np.load("data/processed/y_hodge.npy").astype(np.int64)

X_img = X_enhanced[:, :180].reshape(-1, 1, 12, 15)
X_scalar = X_enhanced[:, 180:]

# Load model
model = CICYClassifier()
model.load_state_dict(
    torch.load("models/cicy_cnn_v1.pt", map_location="cpu")
)
model.eval()

# Prediction
with torch.no_grad():
    img_tensor = torch.from_numpy(X_img)
    scalar_tensor = torch.from_numpy(X_scalar)

    out_h11, out_h21 = model(img_tensor, scalar_tensor)

    pred_h11 = torch.argmax(out_h11, dim=1).numpy()
    pred_h21 = torch.argmax(out_h21, dim=1).numpy()

# Exact accuracy
true_h11 = y_actual[:, 0]
true_h21 = y_actual[:, 1]

total = len(y_actual)

acc_h11 = np.mean(pred_h11 == true_h11) * 100
acc_h21 = np.mean(pred_h21 == true_h21) * 100


print("-" * 40)
print(f"Total manifolds evaluated : {total}")
print(f"h^{1,1} exact accuracy     : {acc_h11:.2f}%")
print(f"h^{2,1} exact accuracy     : {acc_h21:.2f}%")
print("-" * 40)

# Interpretation
if acc_h11 > 90:
    print("STATUS: h^{1,1} prediction is highly reliable.")
if acc_h21 > 50:
    print("STATUS: h^{2,1} prediction shows non-trivial learning.")

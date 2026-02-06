import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from src.models.cnn_model import CICYClassifier

# 1. Load data and model
# Ensure we are using X_enhanced for the 181-feature input
X_test_all = np.load('data/processed/X_enhanced.npy').astype(np.float32)
y_actual = np.load('data/processed/y_hodge.npy').astype(np.float32)

model = CICYClassifier()
model.load_state_dict(torch.load('models/cicy_cnn_v1.pt'))
model.eval()

# 2. Predict using Argmax
with torch.no_grad():
    # Split the enhanced data back into image and scalar
    X_img = torch.from_numpy(X_test_all[:, :180].reshape(-1, 1, 12, 15))
    X_scalar = torch.from_numpy(X_test_all[:, 180:])
    
    # Get raw logit outputs from the two heads
    out_h11, out_h21 = model(X_img, X_scalar)
    
    # Pick the index with the highest score (the predicted Hodge number)
    pred_h11 = torch.argmax(out_h11, dim=1).numpy()
    pred_h21 = torch.argmax(out_h21, dim=1).numpy()

# 3. Calculate Exact Accuracy
total = len(y_actual)
correct_h11 = (pred_h11 == y_actual[:, 0]).sum()
correct_h21 = (pred_h21 == y_actual[:, 1]).sum()

print("-" * 30)
print(f"Total Manifolds Tested: {total}")
print(f"h1,1 Exact Accuracy: {100 * correct_h11 / total:.2f}%")
print(f"h2,1 Exact Accuracy: {100 * correct_h21 / total:.2f}%")
print("-" * 30)

# 4. Success Check
if (100 * correct_h11 / total) > 90:
    print("STATUS: h1,1 Prediction is PhD-level ready!")
if (100 * correct_h21 / total) > 50:
    print("STATUS: h2,1 Prediction is significantly improved!")
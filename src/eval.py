'''
Evaluation script for the CICY CNN model.
Loads the test data and the trained model, runs inference,
and plots the predicted vs actual Hodge numbers.
'''

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.cnn_model import CICYClassifier

# 1. Load Data and Model
X_test = np.load('data/processed/X_enhanced.npy').astype(np.float32)    
y_test = np.load('data/processed/y_hodge.npy').astype(np.float32)

model = CICYClassifier()
model.load_state_dict(torch.load('models/cicy_cnn_v1.pt'))
model.eval()


# 2. Run Inference
with torch.no_grad():
    # Split the enhanced data back into image and scalar
    img_tensor = torch.from_numpy(X_test[:, :180].reshape(-1, 1, 12, 15))
    scalar_tensor = torch.from_numpy(X_test[:, 180:])
    
    # Get raw logit outputs from the two heads
    out_h11, out_h21 = model(img_tensor, scalar_tensor)
    
    # Use argmax to get the predicted class index (the Hodge number)
    pred_h11 = torch.argmax(out_h11, dim=1).numpy()
    pred_h21 = torch.argmax(out_h21, dim=1).numpy()

# 3. Plotting for h11 and h21
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot h11
ax1.scatter(y_test[:, 0], pred_h11, alpha=0.3, color='blue')
ax1.plot([y_test[:, 0].min(), y_test[:, 0].max()], [y_test[:, 0].min(), y_test[:, 0].max()], 'r--')
ax1.set_title('$h^{1,1}$ Prediction')
ax1.set_xlabel('Actual')
ax1.set_ylabel('Predicted')

# Plot h21
ax2.scatter(y_test[:, 1], pred_h21, alpha=0.3, color='green')
ax2.plot([y_test[:, 1].min(), y_test[:, 1].max()], [y_test[:, 1].min(), y_test[:, 1].max()], 'r--')
ax2.set_title('$h^{2,1}$ Prediction')
ax2.set_xlabel('Actual')
ax2.set_ylabel('Predicted')

plt.tight_layout()
plt.savefig('plots/results_v1.png')
plt.show()
'''
Enhance the dataset by adding a new feature: the sum of ambient dimensions for each manifold.
This feature captures the complexity of the ambient space in which the manifold is embedded,
which may correlate with the Hodge numbers.
'''

import numpy as np

# Load raw X and y
X = np.load('data/processed/X_cicy3.npy')
y = np.load('data/processed/y_hodge.npy')

# Calculate the ambient dimension for each manifold
# In CICY, this is the sum of the projective space dimensions
# For our padded matrix, it's roughly the sum of non-zero rows
amb_dim = np.sum(X > 0, axis=(1, 2)) 

# Flatten X and append this new feature
X_flat = X.reshape(len(X), -1)
X_enhanced = np.hstack((X_flat, amb_dim.reshape(-1, 1)))

# Save the enhanced dataset
np.save('data/processed/X_enhanced.npy', X_enhanced)
print("Enhanced dataset created with Ambient Dimension feature!")
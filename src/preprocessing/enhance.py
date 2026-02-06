import numpy as np

# Load raw X and y
X = np.load('data/processed/X_cicy3.npy')
y = np.load('data/processed/y_hodge.npy')

# Compute the 'sum of ambient dimensions' for each manifold
# In CICY, this is the sum of the projective space dimensions
# For our padded matrix, it's roughly the sum of non-zero rows
amb_dim = np.sum(X > 0, axis=(1, 2)) 

# Flatten X and append this new feature (Feature Engineering)
X_flat = X.reshape(len(X), -1)
X_enhanced = np.hstack((X_flat, amb_dim.reshape(-1, 1)))

np.save('data/processed/X_enhanced.npy', X_enhanced)
print("Enhanced dataset created with Ambient Dimension feature!")
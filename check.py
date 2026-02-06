import numpy as np

y = np.load('data/processed/y_hodge.npy')
print(f"Unique h1,1 values: {len(np.unique(y[:, 0]))}")
print(f"Unique h2,1 values: {len(np.unique(y[:, 1]))}")
print(f"Max h1,1: {y[:, 0].max()}, Max h2,1: {y[:, 1].max()}")
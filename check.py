"""
Sanity checks for processed CICY Hodge number data.
Reports label ranges and cardinalities.
"""

import numpy as np


y = np.load("data/processed/y_hodge.npy").astype(np.int64)

h11 = y[:, 0]
h21 = y[:, 1]

print("-" * 40)
print(f"Number of samples      : {len(y)}")
print(f"Unique h^{1,1} values  : {len(np.unique(h11))}")
print(f"Unique h^{2,1} values  : {len(np.unique(h21))}")
print(f"Max h^{1,1}            : {h11.max()}")
print(f"Max h^{2,1}            : {h21.max()}")
print("-" * 40)

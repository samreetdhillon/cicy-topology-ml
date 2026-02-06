import numpy as np
import re
import os

def parse_cicy3_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    all_matrices = []
    all_hodge = []
    
    current_matrix = []
    current_h11 = None
    current_h21 = None

    for line in lines:
        line = line.strip()
        
        # 1. Extract Hodge Numbers
        if line.startswith('H11'):
            current_h11 = int(line.split(':')[-1])
        if line.startswith('H21'):
            current_h21 = int(line.split(':')[-1])
            
        # 2. Extract Matrix Rows (lines starting with '{')
        # We ignore the 'C2' and 'Redun' lines which also use braces
        if line.startswith('{') and not any(x in line for x in ['C2', 'Redun']):
            # Convert "{1, 1, 0...}" to [1, 1, 0...]
            row = [int(x) for x in re.findall(r'\d+', line)]
            current_matrix.append(row)
            
        # 3. Detect end of entry (empty line) and save
        if line == "" and current_matrix:
            all_matrices.append(current_matrix)
            all_hodge.append([current_h11, current_h21])
            # Reset for next entry
            current_matrix = []
            current_h11, current_h21 = None, None

    return all_matrices, np.array(all_hodge)

def pad_matrices(matrices, max_rows=12, max_cols=15):
    """
    CNNs need fixed input sizes. We pad smaller matrices with zeros.
    """
    padded_X = np.zeros((len(matrices), max_rows, max_cols))
    for i, mat in enumerate(matrices):
        m = np.array(mat)
        r, c = m.shape
        padded_X[i, :r, :c] = m
    return padded_X

if __name__ == "__main__":
    raw_path = 'data/raw/cicy3folds.txt'
    mats, hodge = parse_cicy3_file(raw_path)
    X = pad_matrices(mats)
    
    np.save('data/processed/X_cicy3.npy', X)
    np.save('data/processed/y_hodge.npy', hodge)
    print(f"Successfully processed {len(X)} manifolds.")
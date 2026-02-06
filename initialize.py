import os

def create_structure():
    folders = [
        'data/raw',         # Where your .txt file goes
        'data/processed',   # For your cleaned .npy or .csv files
        'models',           # To save your trained .pt or .h5 weights
        'notebooks',        # For exploratory data analysis (EDA)
        'src/preprocessing',# Scripts to turn .txt into tensors
        'src/models',       # Your CNN/MLP architectures
        'plots'             # For loss curves and saliency maps
    ]
    
    files = {
        'requirements.txt': '',
        'README.md': '# CICY Topology ML\nPredicting Hodge numbers using deep learning.',
        'src/__init__.py': '',
        'src/train.py': '# Training loop script',
        '.gitignore': 'data/\nmodels/\n__pycache__/\n.ipynb_checkpoints/'
    }

    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")

    for file_path, content in files.items():
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Created file: {file_path}")

if __name__ == "__main__":
    create_structure()
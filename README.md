# CICY Topology ML

A PyTorch project for predicting topological quantities (Hodge numbers) from CICY (Complete Intersection Calabi-Yau) dataset representations using a convolutional neural network.

## Contents

- `src/` — training and model code
- `data/` — dataset (raw and processed)
- `models/` — saved model weights
- `notebooks/` — exploratory notebooks
- `plots/` — visualization outputs

## Quickstart

1. Create a Python virtual environment and activate it:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Prepare data
   - Place raw data under `data/raw/` as appropriate. Processed arrays used by the training script are expected at `data/processed/X_cicy3.npy` and `data/processed/y_hodge.npy`.

4. Run training

   ```powershell
   python src\train.py
   ```

## Project structure

- `src/train.py` — main training script (loads `data/processed/*.npy`, creates dataloaders, trains model, saves weights to `models/`)
- `src/models/cnn_model.py` — model definition used by the trainer
- `src/preprocessing/parser.py` — (optional) scripts to parse and preprocess raw dataset into NumPy arrays

## Configuration

- Hyperparameters (batch size, learning rate, epochs) are currently defined in `src/train.py`. For experiments, consider refactoring to use a config file or CLI arguments.

## Data format

- `X_cicy3.npy`: float32 array of input features (shape should match model input)
- `y_hodge.npy`: float32 array of target values (Hodge numbers)

## Saving and loading models

- Trained weights are saved by default to `models/cicy_cnn_v1.pt`.
- To load weights in code:

```python
from src.models.cnn_model import CICYCNN
model = CICYCNN()
model.load_state_dict(torch.load('models/cicy_cnn_v1.pt'))
model.eval()
```

## Development notes

- When running `python src/train.py` from the project root, imports use the `src` package; the trainer script adds the project root to `sys.path` to facilitate imports. Consider installing the package in editable mode during development:

```powershell
pip install -e .
```

## Common issues

- Module import errors: ensure you run commands from the project root or use an editable install.
- Missing data files: ensure `data/processed/X_cicy3.npy` and `data/processed/y_hodge.npy` exist prior to running training.

## Contributing

PRs are welcome. Open issues for bugs or feature requests. Include reproducible steps and small, focused changes.

## License

Specify a license for the project by adding a `LICENSE` file. If none is present, contact the maintainer for guidance.

## Contact

Project owner: Samreet (local repository). Add contact details here if you want collaborators to reach you.

# CICY Topology ML

Predicting Hodge numbers using deep learning.

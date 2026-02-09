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

- `src/preprocessing/parser.py` — an optional scripts to parse and preprocess raw dataset into NumPy arrays
- `src/train.py` — main training script (loads `data/processed/*.npy`, creates dataloaders, trains model, saves weights to `models/`)
- `src/models/cnn_model.py` — model definition used by the trainer

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

## Common issues

- **Module import errors**: ensure you run commands from the project root or use an editable install.
- **Missing data files\*\***: ensure `data/processed/X_cicy3.npy` and `data/processed/y_hodge.npy` exist prior to running training.

## Contributing

PRs are welcome. Open issues for bugs or feature requests. Include reproducible steps and small, focused changes.

## Contact

Samreet Singh Dhillon, \
M.Sc. Physics, Panjab Univerity
samreetsinghdhillon@gmail.com

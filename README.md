# OurTorch (`my_torch`)

OurTorch is a lightweight deep learning framework built from scratch in Python and NumPy (`mytorch`), paired with a chess board evaluation and state classification system (`my_torch_analyzer` and `my_torch_generator`).

---

## Project Overview

The project is split into three main components:
1. **`mytorch`**: A modular neural network library inspired by PyTorch, providing weight initializers, layer abstractions, activations, loss functions, optimizers, and model serialization.
2. **`analyzer` (`my_torch_analyzer`)**: The command-line application that loads neural networks and chess datasets in FEN (Forsyth–Edwards Notation) format to either train a model (`--train`) or evaluate/predict outcomes on chessboards (`--predict`).
3. **`scripts` (`my_torch_generator`)**: A utility executable used to generate new neural network architectures from configuration files.

---

## Project Status

| Component | Status | Issue |
| :--- | :--- | :--- |
| Project structure, Makefile, error handler | ✅ Done | #9 |
| `my_torch_analyzer` argument parser | ✅ Done | #7 |
| Weight initializers & base `Layer` | ✅ Done | #8 |
| `Linear`, activations (ReLU, LeakyReLU, Sigmoid, Tanh, Softmax), `Dropout` / MC Dropout | ✅ Done | #5 |
| Losses (MSE, cross-entropy), optimizers (SGD, Adam), `Sequential` model with early stopping | ✅ Done | #4 |
| Input normalization, `.conf` parser, `.nn` save/load, `my_torch_generator`, basic/medium/full configs | ✅ Done | #3 |
| FEN parser & feature encoding (773 features) | ✅ Done | #2 |
| `--train` / `--predict` pipelines | ⏳ To do | #1 |
| Benchmarks, pre-trained networks, docs | ⏳ To do | #6 |

### Building a network from a `.conf` file

```python
from mytorch import build_from_config

model = build_from_config("configs/medium_network.conf")  # validated, seeded, blank network
history = model.fit(X, y, **model.training_config)      # epochs, batch_size, val_split, patience from the file
```

The `.conf` format (JSON) and every validation rule are described in [ARCHITECTURE.md](./ARCHITECTURE.md#mytorchconfigpy--network-configuration-files-conf).

### Using the `mytorch` library (preview)

```python
import numpy as np
from mytorch import Adam, Sequential, SoftmaxCrossEntropyLoss
from mytorch.layers import Dropout, Linear, ReLU

np.random.seed(42)                      # reproducible weights, shuffling and dropout masks
model = Sequential(
    layers=[Linear(773, 256, init="he"), ReLU(), Dropout(0.2), Linear(256, 3, init="xavier")],
    loss=SoftmaxCrossEntropyLoss(),     # the network outputs raw logits: no Softmax layer
    optimizer=Adam(lr=0.001, weight_decay=1e-4),
    classes=["Nothing", "Check", "Checkmate"],
)

X = np.random.rand(1000, 773).astype(np.float32)           # FEN features (encode_fens)
y = np.eye(3, dtype=np.float32)[np.random.randint(0, 3, 1000)]  # one-hot labels, same order as classes

history = model.fit(X, y, epochs=30, batch_size=128, val_split=0.15, patience=5, verbose=True)
print(history["best_epoch"], history["val_acc"][-1])
print(model.predict(X[:5]))             # ['Check', 'Nothing', ...]
```

See [ARCHITECTURE.md](./ARCHITECTURE.md#32-neural-network-library-mytorch) for formulas, conventions, the training loop and gradient checking.

---

## Quick Start

### 1. Build the Executables
Use the `Makefile` at the root of the repository to generate the executable binaries:

```bash
make
```

This creates two executable scripts in the project root:
- `./my_torch_analyzer`
- `./my_torch_generator`

> ⚠️ The executables use `#!/usr/bin/env python3`: **NumPy must be installed for that interpreter** (`pip install numpy`), or the virtual environment must be activated (`source .venv/bin/activate`). Otherwise `my_torch_generator` stops with `Error: Cannot load the mytorch library` and exit code 84.

### 2. Clean Build Artifacts
```bash
# Remove Python cache files (__pycache__, .pytest_cache, *.pyc)
make clean

# Remove cache files AND executable binaries
make fclean

# Rebuild everything from scratch
make re
```

---

## Usage

### `my_torch_generator`

Generate blank (untrained) networks from configuration files:

```bash
./my_torch_generator configs/basic_network.conf 3 configs/full_network.conf 1
# basic_network_1.nn
# basic_network_2.nn
# basic_network_3.nn
# full_network_1.nn
```

Files are written in the current directory. With a `seed` in the `.conf`, the networks are all different but the same command always produces the same files. See [ARCHITECTURE.md](./ARCHITECTURE.md#scriptsgeneratorpy--my_torch_generator) for details and the `.nn` file format.

### `my_torch_analyzer`

Display help:
```bash
./my_torch_analyzer --help
```

```text
USAGE
    ./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE

DESCRIPTION
    --train     Launch the neural network in training mode. Each chessboard in FILE must contain inputs to send to the neural network in FEN notation and the expected output separated by space. If specified, the newly trained neural network will be saved in SAVEFILE. Otherwise, it will be saved in the original LOADFILE.
    --predict   Launch the neural network in prediction mode. Each chessboard in FILE must contain inputs to send to the neural network in FEN notation, and optionally an expected output.
    --save      Save neural network into SAVEFILE. Only works in train mode.
    LOADFILE    File containing an artificial neural network
    CHESSFILE   File containing chessboards
```

#### Examples
```bash
# Run predictions using a pre-trained network on a file of FEN chessboards
./my_torch_analyzer --predict model.nn chessboards.txt

# Train a network in-place (overwrites model.nn)
./my_torch_analyzer --train model.nn training_boards.txt

# Train a network and save the updated weights to a new file
./my_torch_analyzer --train --save trained_model.nn model.nn training_boards.txt
```

### Error Handling Convention
All invalid CLI arguments, missing/unreadable files, or runtime errors write a formatted message (`Error: <message>`) to `stderr` and exit with Epitech error code **`84`**.

---

## Running Tests

Run the unit test suite using `pytest` (or Python's built-in `unittest`):

```bash
# Using pytest inside the virtual environment
.venv/bin/pytest tests/ -v

# Or using unittest
.venv/bin/python3 -m unittest discover -s tests -v
```

---

## Documentation Index

- **[ARCHITECTURE.md](./ARCHITECTURE.md)**: Detailed directory structure, module breakdown, mathematical specifications for initializers, and design patterns used across `mytorch` and `analyzer`.
- **[docs/](./docs/README.md)** *(French)*: complete documentation of every component, for the team and the defense:

  | Chapter | Content |
  | :--- | :--- |
  | [01 — Vue d'ensemble](./docs/01-vue-d-ensemble.md) | Problem, components, network life cycle, step-by-step prediction |
  | [02 — Théorie](./docs/02-theorie.md) | Matrices, activations, softmax, loss, gradient descent, backpropagation, initialization, overfitting |
  | [03 — Les couches](./docs/03-couches.md) | `Layer` contract, initializers, `Linear`, activations, `Dropout`, `Normalization` |
  | [04 — Entraînement](./docs/04-entrainement.md) | Losses, optimizers, `Sequential` and `fit()` |
  | [05 — Configuration et sauvegarde](./docs/05-configuration-et-sauvegarde.md) | `.conf` format, `.nn` format, `my_torch_generator` |
  | [06 — Analyseur et outils](./docs/06-analyseur-et-outils.md) | `my_torch_analyzer` CLI, error handling, `Makefile` |
  | [07 — Tests](./docs/07-tests.md) | Test map, gradient check, writing tests |
  | [08 — Soutenance](./docs/08-soutenance.md) | Likely questions and answers, compliance with the subject, known limits, demos |
  | [09 — Guide d'utilisation](./docs/09-guide-utilisation.md) | How to use `mytorch`: ready-to-run examples, recipes, common errors, API cheat sheet |

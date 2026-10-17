# OurTorch (`my_torch`)

OurTorch is a lightweight deep learning framework built from scratch in Python and NumPy (`mytorch`), paired with a chess board evaluation and state classification system (`my_torch_analyzer` and `my_torch_generator`).

---

## Project Overview

The project is split into three main components:
1. **`mytorch`**: A modular neural network library inspired by PyTorch, providing weight initializers, layer abstractions, activations, loss functions, optimizers, and model serialization.
2. **`analyzer` (`my_torch_analyzer`)**: The command-line application that loads neural networks and chess datasets in FEN (Forsyth–Edwards Notation) format to either train a model (`--train`) or evaluate/predict outcomes on chessboards (`--predict`).
3. **`scripts` (`my_torch_generator`)**: A utility executable used to generate new neural network architectures from configuration files.

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

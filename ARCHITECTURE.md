# Project Structure & Architecture

This document explains the directory layout, architectural design, and core modules of **OurTorch**.

---

## 1. Directory Structure

```text
OurTorch/
├── Makefile                    # Build system for my_torch_analyzer & my_torch_generator
├── README.md                   # General project overview and usage guide
├── ARCHITECTURE.md             # Detailed structure and technical explanations
├── .gitignore                  # Git ignore rules for caches, venvs, and binaries
│
├── mytorch/                    # Custom lightweight neural network library
│   ├── __init__.py             # Public API (Sequential, losses, optimizers)
│   ├── initializers.py         # Weight initialization strategies (Xavier, He, LeCun)
│   ├── losses.py               # MSELoss, SoftmaxCrossEntropyLoss, get_loss()
│   ├── optimizers.py           # SGD (momentum), Adam, get_optimizer()
│   ├── model.py                # `Sequential` model: training loop, early stopping, predict
│   └── layers/                 # Neural network layers package
│       ├── __init__.py         # Exposes Layer and every concrete layer
│       ├── base.py             # Abstract base class `Layer`
│       ├── dense.py            # `Linear` (fully connected) layer
│       ├── activations.py      # ReLU, LeakyReLU, Sigmoid, Tanh, Softmax
│       └── regularization.py   # `Dropout` / MC Dropout
│
├── analyzer/                   # Chessboard analyzer application
│   ├── __init__.py             # Package initialization
│   ├── main.py                 # Entrypoint copied to ./my_torch_analyzer
│   ├── cli.py                  # Strict CLI parser & argument validator
│   ├── fen_encoder.py          # FEN parsing/validation, input encoding, label mapping
│   └── utils.py                # Central error handling (`exit_error`)
│
├── scripts/                    # Standalone scripts and tools
│   └── generator.py            # Entrypoint copied to ./my_torch_generator
│
└── tests/                      # Unit test suite
    ├── __init__.py             # Test path configuration
    ├── gradcheck.py            # Reusable numerical gradient checker (finite differences)
    ├── test_cli.py             # Unit tests for CLI parsing and exit code 84 rules
    ├── test_fen_encoder.py     # FEN parsing on the subject boards, encoding and label mapping
    ├── test_initializers.py    # Unit tests for initializers and base Layer
    ├── test_layers.py          # Gradient checks and behaviour tests for every layer
    ├── test_losses.py          # Loss values, stability and gradient checks
    ├── test_optimizers.py      # Update rules, convergence and config factory
    └── test_xor.py             # Sequential: XOR acceptance test, early stopping, reproducibility
```

---

## 2. Build System (`Makefile`)

Because Python is an interpreted language, the `Makefile` packages the entrypoints as standalone executables using standard POSIX permissions and the `#!/usr/bin/env python3` shebang at the top of each entrypoint:

| Rule | Action |
| :--- | :--- |
| `make` / `make all` | Copies `analyzer/main.py` $\rightarrow$ `./my_torch_analyzer` and `scripts/generator.py` $\rightarrow$ `./my_torch_generator`, and runs `chmod +x` on both. |
| `make clean` | Removes `__pycache__`, `.pytest_cache`, and `*.pyc` files across the workspace. |
| `make fclean` | Runs `clean` and deletes `./my_torch_analyzer` and `./my_torch_generator`. |
| `make re` | Runs `fclean` followed by `all`. |

---

## 3. Core Modules Breakdown

### 3.1 Analyzer Package (`analyzer/`)

#### `analyzer/utils.py`
Provides centralized error reporting required by the specification:
- **`exit_error(message: str) -> None`**:
  Writes `f"Error: {message}\n"` to `sys.stderr` and terminates the process with `sys.exit(84)`.

#### `analyzer/cli.py`
Handles command-line parsing without relying on `argparse`'s default exit code (`2`), ensuring every validation failure exits with code `84`.
- **`AnalyzerArgs`**: A `@dataclass` holding the validated parameters:
  - `mode: str` (`"train"` or `"predict"`)
  - `loadfile: str` (path to the neural network file)
  - `chessfile: str` (path to the FEN dataset file)
  - `savefile: Optional[str]` (optional output path when `--train` is active)
- **Validation Rules**:
  1. `-h` or `--help` prints the exact specification usage to `stdout` and exits with `0`.
  2. Exactly one mode (`--train` or `--predict`) must be specified.
  3. `--save SAVEFILE` is only permitted with `--train` (using `--save` with `--predict` exits with `84`).
  4. Both `LOADFILE` and `CHESSFILE` must exist, be regular files, be readable, and be non-empty (`size > 0`).

#### `analyzer/fen_encoder.py`
Turns `CHESSFILE` lines into network inputs and targets. Every invalid input raises `ValueError` (mapped to exit code `84` by the analyzer).
- **`parse_chess_line(line, require_label)`**: splits a line into `(fen, label)`. The FEN has 6 fields, or 4 when the move counters are omitted; the rest of the line is the label (`"Nothing"`, `"Check White"`, `"Checkmate Black"`, ...). In train mode the label is mandatory and must be known; in predict mode it is optional and not validated.
- **`validate_fen(fen)`**: 8 ranks of exactly 8 squares, pieces in `PNBRQKpnbrqk`, active color `w`/`b`, castling `-` or a subset of `KQkq` in order, en passant `-` or a rank 3/6 square, non-negative move counters.
- **`read_chess_file(path, require_label)`**: parses every non-empty line, keeping the file order; errors mention `path:line`.
- **`encode_fen(fen)` / `encode_fens(fens)`**: float32 vector of **773** values:

| Indices | Content |
| :--- | :--- |
| `0 – 767` | One-hot planes: `piece * 64 + square`, pieces in `PNBRQKpnbrqk` order, square `0 = a8 … 63 = h1` (FEN reading order) |
| `768` | Side to move (`1.0` = White) |
| `769 – 772` | Castling rights `K`, `Q`, `k`, `q` |

  The encoding deliberately contains **no hand-crafted chess features** (attacked king, number of attackers, escape squares, ...): computing them is already a rule-based check detector, which the subject forbids as a predictor. The network must learn these patterns from the raw board.
- **`adapt_label(label, classes)` / `encode_labels(labels, classes)`**: maps dataset labels onto the classes of the loaded network, then one-hot encodes them in `classes` order. Finer labels are coarsened when needed: `Checkmate White → Checkmate → Check`, `Check Black → Check`, `Stalemate → Nothing`. A coarse label (`"Check"`) cannot fill finer classes (`"Check White"`) and raises.
- **Label colors** follow the subject's example: the color is the side that *gives* check or mate (White to move and mated → `"Checkmate Black"`).

---

### 3.2 Neural Network Library (`mytorch/`)

#### `mytorch/initializers.py`
Implements standard weight initialization strategies to prevent vanishing or exploding gradients during forward and backward propagation. All functions accept `shape: tuple[int, int]` representing `(n_in, n_out)` and return a `np.ndarray` of dtype `np.float32`.

| Function | Formula / Distribution | Typical Use Case |
| :--- | :--- | :--- |
| `xavier_normal(shape)` | $\mathcal{N}\!\left(0, \sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}}\right)$ | Tanh / Sigmoid / Logistic activations |
| `xavier_uniform(shape)` | $\mathcal{U}\!\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)$ | Tanh / Sigmoid / Logistic activations |
| `he_normal(shape)` | $\mathcal{N}\!\left(0, \sqrt{\frac{2}{n_{\text{in}}}}\right)$ | ReLU and Leaky ReLU activations |
| `lecun_normal(shape)` | $\mathcal{N}\!\left(0, \sqrt{\frac{1}{n_{\text{in}}}}\right)$ | SELU / standard normalizing networks |

- **`get_initializer(name: str)`**: Factory lookup that normalizes the initializer name (case-insensitive) and raises a `ValueError` if an unknown name is requested. Supported aliases:
  - short names used in `.conf` files: `he`, `kaiming` → `he_normal`; `xavier`, `glorot` → `xavier_normal`; `lecun` → `lecun_normal`;
  - long names: `glorot_normal`, `glorot_uniform`, `kaiming_normal`.

#### `mytorch/layers/base.py`
Defines the abstract base class **`Layer(ABC)`**, which forms the contract for all layers (Dense/Linear, Activation layers, Dropout, BatchNorm, etc.):
- **`self.training: bool = True`**: Tracks whether the layer is in training mode or inference/evaluation mode (crucial for layers like Dropout or Batch Normalization).
- **`forward(self, inputs: np.ndarray) -> np.ndarray`** *(abstract)*: Computes layer output from input activations.
- **`backward(self, grad_output: np.ndarray) -> np.ndarray`** *(abstract)*: Computes gradient with respect to inputs ($\frac{\partial L}{\partial X}$) and accumulates parameter gradients ($\frac{\partial L}{\partial W}, \frac{\partial L}{\partial b}$).
- **`parameters(self) -> list[tuple[np.ndarray, np.ndarray]]`**: Returns pairs of `(parameter_array, gradient_array)` for optimizers to update. Returns `[]` by default for parameter-free layers.
- **`get_config(self) -> dict`**: Returns a dictionary describing the layer's configuration for saving/serializing models.

#### Layer conventions (read this before writing a new layer)
- Every tensor is 2D: `(batch, features)`. One row = one sample.
- `forward()` caches what `backward()` needs. Calling `backward()` before `forward()` raises `RuntimeError`.
- `backward(grad_output)` receives $\frac{\partial L}{\partial Y}$ and returns $\frac{\partial L}{\partial X}$. Parameter gradients are **overwritten** (not accumulated) at each call.
- Parameter gradients are allocated once and written **in place** (`self.dW[...] = ...`), so the `(param, grad)` references returned by `parameters()` remain valid for the optimizer across training steps. Never rebind `self.dW = ...` in `backward()`.
- `get_config()` returns exactly the dictionary format of a layer entry in the `.conf` files, so serialization (`save_network` / `build_from_config`, issue #3) can rebuild any layer from it.

#### `mytorch/layers/dense.py` — `Linear(in_features, out_features, init="he")`
| Step | Formula | Shapes |
| :--- | :--- | :--- |
| Forward | $Y = XW + b$ | $X: (N, n_{in})$, $W: (n_{in}, n_{out})$, $b: (1, n_{out})$ |
| Backward (weights) | $\frac{\partial L}{\partial W} = X^\top G$ | $(n_{in}, n_{out})$ |
| Backward (bias) | $\frac{\partial L}{\partial b} = \sum_{rows} G$ | $(1, n_{out})$ |
| Backward (input) | $\frac{\partial L}{\partial X} = G W^\top$ | $(N, n_{in})$ |

- `W` is initialized with `get_initializer(init)`, `b` with zeros. Both are `float32`.
- Raises `ValueError` on non-positive dimensions, unknown initializer, or wrong input width.
- Config: `{"type": "linear", "in": 789, "out": 256, "init": "he"}`.

#### `mytorch/layers/activations.py`
All activations are parameter-free (`parameters() == []`) and inherit from a small `Activation` helper that stores the forward cache.

| Layer | Forward | Backward ($G$ = upstream gradient) | Config |
| :--- | :--- | :--- | :--- |
| `ReLU()` | $\max(0, x)$ | $G \cdot [x > 0]$ | `{"type": "relu"}` |
| `LeakyReLU(alpha=0.01)` | $x$ if $x>0$ else $\alpha x$ | $G \cdot (1 \text{ or } \alpha)$ | `{"type": "leakyrelu", "alpha": 0.01}` |
| `Sigmoid()` | $\sigma(x) = 1/(1+e^{-\text{clip}(x,-50,50)})$ | $G \cdot \sigma(1-\sigma)$ | `{"type": "sigmoid"}` |
| `Tanh()` | $\tanh(x)$ | $G \cdot (1-\tanh^2 x)$ | `{"type": "tanh"}` |
| `Softmax()` | $s_i = e^{x_i - \max x} / \sum_j e^{x_j - \max x}$ (row-wise) | $s \odot (G - \langle G, s\rangle)$ | `{"type": "softmax"}` |

> ⚠️ **Do not put `Softmax` as the last layer when training with `SoftmaxCrossEntropyLoss`** (issue #4): that loss applies softmax itself on raw logits (more stable, and its gradient simplifies to $\hat{Y} - Y$). The network should output raw logits; `Softmax` is only useful to turn logits into probabilities.

#### `mytorch/layers/regularization.py` — `Dropout(rate=0.2, mc_dropout=False)`
Anti-overfitting measure required by **[OPT1]**, plus MC dropout from **[OPT3]**.
- **Inverted dropout**: in training mode each unit is set to 0 with probability `rate`, survivors are multiplied by $\frac{1}{1-\text{rate}}$. The expected activation is unchanged, so inference needs no rescaling.
- **Inference** (`layer.training = False`): the layer is the identity.
- **MC dropout** (`mc_dropout=True`): dropout stays active even in inference mode. Running the same input $T$ times and averaging gives a prediction plus an uncertainty estimate (variance across runs).
- `backward()` reuses the forward mask: dropped units receive no gradient.
- The mask is drawn from NumPy's global RNG → `np.random.seed(...)` makes training reproducible.
- `rate` must be in $[0, 1)$, otherwise `ValueError`. Config: `{"type": "dropout", "rate": 0.2, "mc_dropout": false}`.

#### Gradient checking (`tests/gradcheck.py`)
Every layer with a `backward()` is validated against a numerical gradient:
1. Draw a random upstream gradient $G$ and define the scalar loss $L = \sum (\text{forward}(X) \odot G)$, so $\frac{\partial L}{\partial Y} = G$.
2. Analytic gradients: `backward(G)` (input) and `parameters()` (weights).
3. Numeric gradients: central differences $\frac{L(x+\varepsilon) - L(x-\varepsilon)}{2\varepsilon}$ with $\varepsilon = 10^{-6}$.
4. Relative error $\frac{\lVert a - n \rVert}{\lVert a \rVert + \lVert n \rVert}$ must be $< 10^{-5}$.

Checks run in `float64` (float32 is too imprecise for this tolerance), and ReLU inputs are kept away from 0 where the derivative is undefined. Reuse `check_layer_gradients(layer, X)` when adding a new layer, and `check_loss_gradient(loss, y_pred, y_true)` when adding a new loss.

#### `mytorch/losses.py`
A loss exposes `forward(y_pred, y_true) -> float` (mean over the batch) and `backward() -> dL/dy_pred`, which is passed to the last layer.

| Loss | Name in `.conf` | Forward | Gradient |
| :--- | :--- | :--- | :--- |
| `MSELoss` | `mse` | $\frac{1}{N \cdot K}\sum (\hat{y} - y)^2$ (mean over every element) | $\frac{2(\hat{y} - y)}{N \cdot K}$ |
| `SoftmaxCrossEntropyLoss` | `cross_entropy` | $-\frac{1}{N}\sum Y \log \text{softmax}(z)$ on **raw logits** $z$ | $\frac{\text{softmax}(z) - Y}{N}$ |

- Cross-entropy uses the log-sum-exp trick ($\log\text{softmax}(z) = z - \max z - \log\sum e^{z - \max z}$) instead of clipping probabilities: it never computes `log(0)` and stays exact for extreme logits.
- `output_transform(outputs)` turns raw outputs into predictions: softmax for cross-entropy, identity for MSE. `Sequential.predict_proba()` uses it.
- `get_loss(name)` builds a loss from its `.conf` name (`mse`, `cross_entropy`).

#### `mytorch/optimizers.py`
Optimizers receive the `(param, grad)` list of `Sequential.parameters()` and update `param` **in place**. Their internal state (velocity, Adam moments) is keyed by `id(param)`, which is why layers must never reallocate their parameter arrays.

| Optimizer | Defaults | Update rule |
| :--- | :--- | :--- |
| `SGD` | `lr=0.01, momentum=0.9, weight_decay=0` | $v \leftarrow \mu v - \eta g$ ; $p \leftarrow p + v$ (plain SGD when $\mu = 0$) |
| `Adam` | `lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0` | $m \leftarrow \beta_1 m + (1-\beta_1) g$ ; $v \leftarrow \beta_2 v + (1-\beta_2) g^2$ ; $p \leftarrow p - \eta \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$ with bias correction $\hat{m} = \frac{m}{1-\beta_1^t}$, $\hat{v} = \frac{v}{1-\beta_2^t}$ |

- **L2 regularization ([OPT1])**: `weight_decay` $\lambda$ adds $\lambda p$ to every gradient, i.e. it minimizes $L + \frac{\lambda}{2}\lVert W \rVert^2$. It applies to biases too (kept simple on purpose).
- `lr` is a plain attribute: a learning-rate schedule can change `optimizer.lr` between epochs.
- `get_optimizer({"type": "adam", "lr": 0.001, "weight_decay": 1e-4})` builds an optimizer from a `.conf` entry. Unknown types or parameters raise `ValueError`.

#### `mytorch/model.py` — `Sequential(layers, loss, optimizer, classes)`
Ties everything together. `classes` is the ordered list of output labels (e.g. `["Nothing", "Check", "Checkmate"]`): output neuron $i$ ↔ `classes[i]`, and training targets are one-hot vectors in the same order.

| Method | Role |
| :--- | :--- |
| `train()` / `eval()` | Switch every layer between training mode (Dropout active) and inference mode |
| `forward(X)` / `backward(grad)` | Run the layers in order / in reverse order |
| `parameters()` | Concatenated `(param, grad)` pairs of all layers |
| `train_step(X_batch, y_batch)` | forward → loss → backward → optimizer step; returns the batch loss |
| `fit(X, y, epochs, batch_size, val_split=0.15, patience=5, validation_data=None, verbose=False)` | Full training loop (see below), returns a `history` dict |
| `evaluate(X, y)` | `(mean loss, accuracy)` in inference mode, without changing the current mode |
| `predict_proba(X)` | Outputs after `loss.output_transform` (softmax probabilities), by batches of 1024 |
| `predict(X)` | List of class names (`argmax` of the probabilities) |
| `get_config()` | `{"classes", "loss", "optimizer", "layers"}` — same structure as a `.conf` file (issue #3) |

**`fit()` in detail** (anti-overfitting **[OPT1]**, mini-batch gradient descent **[OPT3]**):
1. **Validation set**: `validation_data=(X_val, y_val)` if given, otherwise a random `val_split` fraction of `(X, y)` is held out. `val_split=0` trains on everything (no validation, no early stopping). For a final score, keep a separate **test set** and pass it to `evaluate()` only at the end.
2. **Each epoch**: training mode, data shuffled, mini-batches of `batch_size`, one optimizer step per batch.
3. **Metrics**: `train_loss` (mean of the batch losses), `train_acc`, `val_loss`, `val_acc` are appended to `history` (to plot learning curves for the benchmarks, issue #6).
4. **Early stopping**: the weights with the best `val_loss` are saved; after `patience` epochs without improvement, training stops and **the best weights are restored**. `history["best_epoch"]` gives the chosen epoch. `patience=None` disables it.
5. The model is left in **inference mode** at the end.

With `verbose=True`, one line per epoch is printed on **stderr**: `stdout` is reserved for the predictions of `my_torch_analyzer --predict`.

**Reproducibility**: shuffling, validation split, weight initialization and dropout masks all use NumPy's global RNG. Calling `np.random.seed(s)` before building and training the model reproduces exactly the same run (tested in `tests/test_xor.py`).

**XOR acceptance test**: `Sequential([Linear(2, 8), ReLU(), Linear(8, 2)])` reaches 100 % accuracy and a loss < 0.02 in 300 epochs, with both Adam (`lr=0.05`) and SGD + momentum (`lr=0.1`). The tests use a fixed seed. Over 50 random seeds, about 88 % of runs succeed: with only 8 hidden units, some initializations leave dead ReLUs. This is expected for such a tiny network, not a bug.

---

## 4. Component Interaction Flow

```mermaid
flowchart LR
    CLI["CLI (analyzer/cli.py)"] --> Main["analyzer/main.py"]
    Main -->|Error| Utils["exit_error() -> Exit 84"]
    Main -->|--train / --predict| Engine["mytorch/model.py (Sequential)"]
    Engine --> Layers["mytorch/layers (Linear, activations, Dropout)"]
    Engine --> Loss["mytorch/losses.py"]
    Engine --> Optim["mytorch/optimizers.py"]
    Layers --> Init["mytorch/initializers.py"]
```

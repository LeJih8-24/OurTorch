# Project Structure & Architecture

This document explains the directory layout, architectural design, and core modules of **OurTorch**.

> A complete, more pedagogical documentation (in French) explaining every component is available in [docs/](./docs/README.md).

---

## 1. Directory Structure

```text
OurTorch/
├── Makefile                    # Build system for my_torch_analyzer & my_torch_generator
├── README.md                   # General project overview and usage guide
├── ARCHITECTURE.md             # Detailed structure and technical explanations
├── docs/                       # Complete documentation in French (one chapter per topic)
├── .gitignore                  # Git ignore rules for caches, venvs, and binaries
│
├── configs/                    # Blank network configurations (.conf, JSON)
│   ├── basic_network.conf      # Nothing / Check
│   ├── medium_network.conf     # Nothing / Check / Checkmate
│   └── full_network.conf       # 5 classes, with the color of the side giving check / mate
│
├── mytorch/                    # Custom lightweight neural network library
│   ├── __init__.py             # Public API (Sequential, losses, optimizers, config)
│   ├── config.py               # .conf validation and build_from_config()
│   ├── initializers.py         # Weight initialization strategies (Xavier, He, LeCun)
│   ├── losses.py               # MSELoss, SoftmaxCrossEntropyLoss, get_loss()
│   ├── optimizers.py           # SGD (momentum), Adam, get_optimizer()
│   ├── model.py                # `Sequential` model: training loop, early stopping, predict
│   ├── serialization.py        # .nn files: save_network(), load_network(), read_metadata()
│   └── layers/                 # Neural network layers package
│       ├── __init__.py         # Exposes Layer and every concrete layer
│       ├── base.py             # Abstract base class `Layer`
│       ├── dense.py            # `Linear` (fully connected) layer
│       ├── activations.py      # ReLU, LeakyReLU, Sigmoid, Tanh, Softmax
│       ├── normalization.py    # `Normalization` (input standardization)
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
│   └── generator.py            # my_torch_generator: blank .nn networks from .conf files
│
└── tests/                      # Unit test suite
    ├── __init__.py             # Test path configuration
    ├── gradcheck.py            # Reusable numerical gradient checker (finite differences)
    ├── test_cli.py             # Unit tests for CLI parsing and exit code 84 rules
    ├── test_fen_encoder.py     # FEN parsing on the subject boards, encoding and label mapping
    ├── test_config.py          # .conf validation, build_from_config, shipped configs
    ├── test_generator.py       # my_torch_generator run as the real executable (exit codes, files)
    ├── test_initializers.py    # Unit tests for initializers and base Layer
    ├── test_layers.py          # Gradient checks and behaviour tests for every layer
    ├── test_losses.py          # Loss values, stability and gradient checks
    ├── test_normalization.py   # Normalization layer, state_dict, integration in fit()
    ├── test_optimizers.py      # Update rules, convergence and config factory
    ├── test_serialization.py   # .nn round trip, corrupted / foreign / truncated files
    └── test_xor.py             # Sequential: XOR acceptance test, early stopping, reproducibility
```

---

## 2. Build System (`Makefile`)

Because Python is an interpreted language, the `Makefile` packages the entrypoints as standalone executables using standard POSIX permissions and the `#!/usr/bin/env python3` shebang at the top of each entrypoint:

| Rule                    | Action                                                                                                                                                                    |
| :---------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `make` / `make all` | Copies `analyzer/main.py` $\rightarrow$ `./my_torch_analyzer` and `scripts/generator.py` $\rightarrow$ `./my_torch_generator`, and runs `chmod +x` on both. |
| `make clean`          | Removes `__pycache__`, `.pytest_cache`, and `*.pyc` files across the workspace.                                                                                     |
| `make fclean`         | Runs `clean` and deletes `./my_torch_analyzer` and `./my_torch_generator`.                                                                                          |
| `make re`             | Runs `fclean` followed by `all`.                                                                                                                                      |

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
Step-by-step walkthrough with encoded examples: [docs/fen_encoder.MD](./docs/fen_encoder.MD).
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

| Function                  | Formula / Distribution                                                                                                          | Typical Use Case                      |
| :------------------------ | :------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------ |
| `xavier_normal(shape)`  | $\mathcal{N}\!\left(0, \sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}}\right)$                                                | Tanh / Sigmoid / Logistic activations |
| `xavier_uniform(shape)` | $\mathcal{U}\!\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)$ | Tanh / Sigmoid / Logistic activations |
| `he_normal(shape)`      | $\mathcal{N}\!\left(0, \sqrt{\frac{2}{n_{\text{in}}}}\right)$                                                                 | ReLU and Leaky ReLU activations       |
| `lecun_normal(shape)`   | $\mathcal{N}\!\left(0, \sqrt{\frac{1}{n_{\text{in}}}}\right)$                                                                 | SELU / standard normalizing networks  |

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
- **`state_keys` / `state_dict()` / `load_state_dict(state)`**: the arrays that must be saved in a `.nn` file besides the configuration. `state_keys` lists attribute names (`("W", "b")` for `Linear`, `("mean", "std", "count")` for `Normalization`, empty for stateless layers). `load_state_dict()` checks keys and shapes (`ValueError` otherwise) and copies the values **in place**.

#### Layer conventions (read this before writing a new layer)

- Every tensor is 2D: `(batch, features)`. One row = one sample.
- `forward()` caches what `backward()` needs. Calling `backward()` before `forward()` raises `RuntimeError`.
- `backward(grad_output)` receives $\frac{\partial L}{\partial Y}$ and returns $\frac{\partial L}{\partial X}$. Parameter gradients are **overwritten** (not accumulated) at each call.
- Parameter gradients are allocated once and written **in place** (`self.dW[...] = ...`), so the `(param, grad)` references returned by `parameters()` remain valid for the optimizer across training steps. Never rebind `self.dW = ...` in `backward()`.
- `get_config()` returns exactly the dictionary format of a layer entry in the `.conf` files, so serialization (`save_network` / `build_from_config`, issue #3) can rebuild any layer from it.

#### `mytorch/layers/dense.py` — `Linear(in_features, out_features, init="he")`

| Step               | Formula                                           | Shapes                                                            |
| :----------------- | :------------------------------------------------ | :---------------------------------------------------------------- |
| Forward            | $Y = XW + b$                                    | $X: (N, n_{in})$, $W: (n_{in}, n_{out})$, $b: (1, n_{out})$ |
| Backward (weights) | $\frac{\partial L}{\partial W} = X^\top G$      | $(n_{in}, n_{out})$                                             |
| Backward (bias)    | $\frac{\partial L}{\partial b} = \sum_{rows} G$ | $(1, n_{out})$                                                  |
| Backward (input)   | $\frac{\partial L}{\partial X} = G W^\top$      | $(N, n_{in})$                                                   |

- `W` is initialized with `get_initializer(init)`, `b` with zeros. Both are `float32`.
- Raises `ValueError` on non-positive dimensions, unknown initializer, or wrong input width.
- Config: `{"type": "linear", "in": 773, "out": 256, "init": "he"}`.

#### `mytorch/layers/activations.py`

All activations are parameter-free (`parameters() == []`) and inherit from a small `Activation` helper that stores the forward cache.

| Layer                     | Forward                                                         | Backward ($G$ = upstream gradient)  | Config                                   |
| :------------------------ | :-------------------------------------------------------------- | :------------------------------------ | :--------------------------------------- |
| `ReLU()`                | $\max(0, x)$                                                  | $G \cdot [x > 0]$                   | `{"type": "relu"}`                     |
| `LeakyReLU(alpha=0.01)` | $x$ if $x>0$ else $\alpha x$                              | $G \cdot (1 \text{ or } \alpha)$    | `{"type": "leakyrelu", "alpha": 0.01}` |
| `Sigmoid()`             | $\sigma(x) = 1/(1+e^{-\text{clip}(x,-50,50)})$                | $G \cdot \sigma(1-\sigma)$          | `{"type": "sigmoid"}`                  |
| `Tanh()`                | $\tanh(x)$                                                    | $G \cdot (1-\tanh^2 x)$             | `{"type": "tanh"}`                     |
| `Softmax()`             | $s_i = e^{x_i - \max x} / \sum_j e^{x_j - \max x}$ (row-wise) | $s \odot (G - \langle G, s\rangle)$ | `{"type": "softmax"}`                  |

> ⚠️ **Do not put `Softmax` as the last layer when training with `SoftmaxCrossEntropyLoss`** (issue #4): that loss applies softmax itself on raw logits (more stable, and its gradient simplifies to $\hat{Y} - Y$). The network should output raw logits; `Softmax` is only useful to turn logits into probabilities.

#### `mytorch/layers/regularization.py` — `Dropout(rate=0.2, mc_dropout=False)`

Anti-overfitting measure required by **[OPT1]**, plus MC dropout from **[OPT3]**.

- **Inverted dropout**: in training mode each unit is set to 0 with probability `rate`, survivors are multiplied by $\frac{1}{1-\text{rate}}$. The expected activation is unchanged, so inference needs no rescaling.
- **Inference** (`layer.training = False`): the layer is the identity.
- **MC dropout** (`mc_dropout=True`): dropout stays active even in inference mode. Running the same input $T$ times and averaging gives a prediction plus an uncertainty estimate (variance across runs).
- `backward()` reuses the forward mask: dropped units receive no gradient.
- The mask is drawn from NumPy's global RNG → `np.random.seed(...)` makes training reproducible.
- `rate` must be in $[0, 1)$, otherwise `ValueError`. Config: `{"type": "dropout", "rate": 0.2, "mc_dropout": false}`.

#### `mytorch/layers/normalization.py` — `Normalization(features)`
Standardizes each input feature: $Y = \frac{X - \mu}{\sigma}$.

**Why**: input features have very different scales (one-hot squares in {0, 1}, king coordinates, number of attackers…). Without standardization, the large features dominate the gradients and training becomes unstable. Measured on a 2-feature toy problem with scales ~1000 and ~0.001 (`tests/test_normalization.py`): SGD diverges to a NaN loss without normalization, and reaches 92 % accuracy with it.

- $\mu$ and $\sigma$ are **not trained**: `adapt(X)` computes them on the training data. `parameters()` is empty, the optimizer never touches them.
- They are part of the layer **state** (`mean`, `std`, `count`), so they are saved in the `.nn` file and prediction applies exactly the same transformation as training.
- Features with $\sigma < 10^{-6}$ on the training set (e.g. a square that is always empty) keep $\sigma = 1$: they are centered, never divided by ~0.
- Before `adapt()`, the layer is the identity. `count` (number of samples used) tells whether it was adapted: `layer.adapted == (count > 0)`.
- Backward: $\frac{\partial L}{\partial X} = \frac{G}{\sigma}$ (gradient-checked).
- **Only allowed as the first layer** (enforced by `Sequential` and the `.conf` validation).
- `Sequential.fit()` adapts it automatically **on the training split only** (never on validation data, which would leak information), and only if it was never adapted: retraining a loaded network keeps its saved statistics.

#### Gradient checking (`tests/gradcheck.py`)

Every layer with a `backward()` is validated against a numerical gradient:

1. Draw a random upstream gradient $G$ and define the scalar loss $L = \sum (\text{forward}(X) \odot G)$, so $\frac{\partial L}{\partial Y} = G$.
2. Analytic gradients: `backward(G)` (input) and `parameters()` (weights).
3. Numeric gradients: central differences $\frac{L(x+\varepsilon) - L(x-\varepsilon)}{2\varepsilon}$ with $\varepsilon = 10^{-6}$.
4. Relative error $\frac{\lVert a - n \rVert}{\lVert a \rVert + \lVert n \rVert}$ must be $< 10^{-5}$.

Checks run in `float64` (float32 is too imprecise for this tolerance), and ReLU inputs are kept away from 0 where the derivative is undefined. Reuse `check_layer_gradients(layer, X)` when adding a new layer, and `check_loss_gradient(loss, y_pred, y_true)` when adding a new loss.

#### `mytorch/losses.py`

A loss exposes `forward(y_pred, y_true) -> float` (mean over the batch) and `backward() -> dL/dy_pred`, which is passed to the last layer.

| Loss                        | Name in `.conf` | Forward                                                                     | Gradient                             |
| :-------------------------- | :---------------- | :-------------------------------------------------------------------------- | :----------------------------------- |
| `MSELoss`                 | `mse`           | $\frac{1}{N \cdot K}\sum (\hat{y} - y)^2$ (mean over every element)       | $\frac{2(\hat{y} - y)}{N \cdot K}$ |
| `SoftmaxCrossEntropyLoss` | `cross_entropy` | $-\frac{1}{N}\sum Y \log \text{softmax}(z)$ on **raw logits** $z$ | $\frac{\text{softmax}(z) - Y}{N}$  |

- Cross-entropy uses the log-sum-exp trick ($\log\text{softmax}(z) = z - \max z - \log\sum e^{z - \max z}$) instead of clipping probabilities: it never computes `log(0)` and stays exact for extreme logits.
- `output_transform(outputs)` turns raw outputs into predictions: softmax for cross-entropy, identity for MSE. `Sequential.predict_proba()` uses it.
- `get_loss(name)` builds a loss from its `.conf` name (`mse`, `cross_entropy`).

#### `mytorch/optimizers.py`

Optimizers receive the `(param, grad)` list of `Sequential.parameters()` and update `param` **in place**. Their internal state (velocity, Adam moments) is keyed by `id(param)`, which is why layers must never reallocate their parameter arrays.

| Optimizer | Defaults                                                       | Update rule                                                                                                                                                                                                                                                   |
| :-------- | :------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `SGD`   | `lr=0.01, momentum=0.9, weight_decay=0`                      | $v \leftarrow \mu v - \eta g$ ; $p \leftarrow p + v$ (plain SGD when $\mu = 0$)                                                                                                                                                                         |
| `Adam`  | `lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0` | $m \leftarrow \beta_1 m + (1-\beta_1) g$ ; $v \leftarrow \beta_2 v + (1-\beta_2) g^2$ ; $p \leftarrow p - \eta \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$ with bias correction $\hat{m} = \frac{m}{1-\beta_1^t}$, $\hat{v} = \frac{v}{1-\beta_2^t}$ |

- **L2 regularization ([OPT1])**: `weight_decay` $\lambda$ adds $\lambda p$ to every gradient, i.e. it minimizes $L + \frac{\lambda}{2}\lVert W \rVert^2$. It applies to biases too (kept simple on purpose).
- `lr` is a plain attribute: a learning-rate schedule can change `optimizer.lr` between epochs.
- `get_optimizer({"type": "adam", "lr": 0.001, "weight_decay": 1e-4})` builds an optimizer from a `.conf` entry. Unknown types or parameters raise `ValueError`.

#### `mytorch/model.py` — `Sequential(layers, loss, optimizer, classes, training_config=None, seed=None)`

Ties everything together. `classes` is the ordered list of output labels (e.g. `["Nothing", "Check", "Checkmate"]`): output neuron $i$ ↔ `classes[i]`, and training targets are one-hot vectors in the same order.

`training_config` (`epochs`, `batch_size`, `val_split`, `patience`) and `seed` come from the `.conf` file and are kept with the model, so a training run uses the configured settings: `model.fit(X, y, **model.training_config)`.

| Method                                                                                             | Role                                                                                            |
| :------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------- |
| `train()` / `eval()`                                                                           | Switch every layer between training mode (Dropout active) and inference mode                    |
| `forward(X)` / `backward(grad)`                                                                | Run the layers in order / in reverse order                                                      |
| `parameters()`                                                                                   | Concatenated `(param, grad)` pairs of all layers                                              |
| `train_step(X_batch, y_batch)`                                                                   | forward → loss → backward → optimizer step; returns the batch loss                           |
| `fit(X, y, epochs, batch_size, val_split=0.15, patience=5, validation_data=None, verbose=False)` | Full training loop (see below), returns a `history` dict                                      |
| `evaluate(X, y)`                                                                                 | `(mean loss, accuracy)` in inference mode, without changing the current mode                  |
| `predict_proba(X)`                                                                               | Outputs after `loss.output_transform` (softmax probabilities), by batches of 1024             |
| `predict(X)`                                                                                     | List of class names (`argmax` of the probabilities)                                           |
| `get_config()` | `{"classes", "loss", "optimizer", "layers", "training", "seed"}` — same structure as a `.conf` file: `build_from_config(model.get_config())` rebuilds the same network |

**`fit()` in detail** (anti-overfitting **[OPT1]**, mini-batch gradient descent **[OPT3]**):

0. **Normalization**: if the first layer is a `Normalization` that was never adapted, its statistics are computed on the training split.
1. **Validation set**: `validation_data=(X_val, y_val)` if given, otherwise a random `val_split` fraction of `(X, y)` is held out. `val_split=0` trains on everything (no validation, no early stopping). For a final score, keep a separate **test set** and pass it to `evaluate()` only at the end.
2. **Each epoch**: training mode, data shuffled, mini-batches of `batch_size`, one optimizer step per batch.
3. **Metrics**: `train_loss` (mean of the batch losses), `train_acc`, `val_loss`, `val_acc` are appended to `history` (to plot learning curves for the benchmarks, issue #6).
4. **Early stopping**: the weights with the best `val_loss` are saved; after `patience` epochs without improvement, training stops and **the best weights are restored**. `history["best_epoch"]` gives the chosen epoch. `patience=None` disables it.
5. The model is left in **inference mode** at the end.

With `verbose=True`, one line per epoch is printed on **stderr**: `stdout` is reserved for the predictions of `my_torch_analyzer --predict`.

**Reproducibility**: shuffling, validation split, weight initialization and dropout masks all use NumPy's global RNG. Calling `np.random.seed(s)` before building and training the model reproduces exactly the same run (tested in `tests/test_xor.py`).

**XOR acceptance test**: `Sequential([Linear(2, 8), ReLU(), Linear(8, 2)])` reaches 100 % accuracy and a loss < 0.02 in 300 epochs, with both Adam (`lr=0.05`) and SGD + momentum (`lr=0.1`). The tests use a fixed seed. Over 50 random seeds, about 88 % of runs succeed: with only 8 hidden units, some initializations leave dead ReLUs. This is expected for such a tiny network, not a bug.

#### `mytorch/config.py` — network configuration files (`.conf`)
A `.conf` file is a **JSON** document describing a blank network. JSON was chosen because it is human-readable, versionable, parsed by the standard library (no extra dependency) and maps 1:1 to `get_config()`.

```json
{
  "classes": ["Nothing", "Check", "Checkmate"],
  "loss": "cross_entropy",
  "optimizer": {"type": "adam", "lr": 0.001, "weight_decay": 0.0001},
  "training": {"epochs": 40, "batch_size": 128, "val_split": 0.15, "patience": 5},
  "seed": 42,
  "layers": [
    {"type": "normalization"},
    {"type": "linear", "in": 773, "out": 256, "init": "he"},
    {"type": "relu"},
    {"type": "dropout", "rate": 0.2},
    {"type": "linear", "in": 256, "out": 3, "init": "xavier"}
  ]
}
```

| Key | Required | Default | Content |
| :--- | :--- | :--- | :--- |
| `classes` | yes | — | Output labels, in the order of the output neurons (≥ 2, unique) |
| `layers` | yes | — | List of layers (table below) |
| `loss` | no | `"cross_entropy"` | `cross_entropy` or `mse` |
| `optimizer` | no | `{"type": "adam"}` | `{"type": "adam" \| "sgd", ...}` + any optimizer parameter (`lr`, `momentum`, `beta1`, `beta2`, `eps`, `weight_decay`) |
| `training` | no | `{"epochs": 30, "batch_size": 128, "val_split": 0.15, "patience": 5}` | `fit()` settings; `patience: null` disables early stopping |
| `seed` | no | `null` | Integer ≥ 0: `np.random.seed(seed)` is called before creating the weights |

| Layer `type` | Parameters (default) |
| :--- | :--- |
| `normalization` | `features` (inferred from the first `linear.in`) — first layer only |
| `linear` | `in`, `out` (required), `init` (`"he"`; also `xavier`, `lecun`, `xavier_uniform`…) |
| `relu`, `sigmoid`, `tanh`, `softmax` | none |
| `leakyrelu` | `alpha` (`0.01`) |
| `dropout` | `rate` (`0.2`), `mc_dropout` (`false`) |

**Validation** (`validate_config(dict)` / `load_config(path)`) is strict, so a typo never silently produces a different network:
- unknown keys (top-level, layer, training) are rejected;
- types and ranges are checked (`bool` is not accepted as an integer);
- consecutive `linear` sizes must match, and the last output size must equal `len(classes)`;
- `softmax` as last layer with `cross_entropy` is rejected (the loss already applies softmax);
- every error is a `ValueError` naming the faulty field, e.g. `layers[4].in is 32 but the previous layer outputs 16 features.` The executables turn it into a message on stderr + exit code 84.

**`build_from_config(path_or_dict)`** validates, seeds NumPy, creates the layers, loss and optimizer, and returns a fresh `Sequential`. `build_from_config(model.get_config())` rebuilds an identical architecture (and identical initial weights when a seed is set).

**Shipped configurations** (`configs/`): `basic_network.conf` (Nothing / Check), `medium_network.conf` (Nothing / Check / Checkmate) and `full_network.conf` (Nothing, Check White/Black, Checkmate White/Black). All start with `normalization` and use dropout + L2 weight decay against overfitting. Their input size **773** is `INPUT_SIZE` of `analyzer/fen_encoder.py`: if the encoder changes, update the first `linear.in` of each file (`test_shipped_configs_match_the_fen_encoder` fails until then).

**Reproducibility note for the pipeline (issue #1)**: `build_from_config` seeds the RNG before initializing the weights. When training a network **loaded** from a file, call `np.random.seed(model.seed)` before `fit()` to get the same shuffling and dropout masks.

#### `mytorch/serialization.py` — network files (`.nn`)
A `.nn` file contains **everything needed to use or retrain a network**: architecture, classes, training settings, weights and input normalization statistics.

**Format**: a compressed NumPy `.npz` archive (a zip of `.npy` arrays):

| Entry | Content |
| :--- | :--- |
| `header` | JSON string `{"format": "mytorch-nn", "version": 1, "metadata": {...}}` |
| `config` | JSON string = `model.get_config()` (same structure as a `.conf` file) |
| `layer{i}.{key}` | One array per layer state entry: `layer1.W`, `layer1.b`, `layer0.mean`, `layer0.std`, `layer0.count`… |

**Why this format** (instead of `pickle`, suggested in the issue):
- **Safe**: loaded with `allow_pickle=False`, a `.nn` file can never execute code, unlike `pickle`.
- **Robust**: independent of the Python class names and module layout, so refactoring `mytorch` does not break old files.
- **Inspectable**: `unzip -l net.nn` lists the arrays, the config is readable JSON.
- **Compact**: weights stay `float32` and the archive is compressed (~1.9 MB for `full_network.conf`).
- **Versioned**: the header `version` lets us change the format later and reject old/new files with a clear error.

**API**
- `save_network(model, path, metadata=None)`: writes the file **atomically** (temporary file in the same directory, then rename). If saving fails, an existing file — for instance `LOADFILE` in `--train` mode — is left intact. `metadata` is a free JSON dict (planned use: FEN encoder version and input size, issue #2).
- `load_network(path) -> Sequential`: rebuilds the network with `build_from_config(config)`, then loads every array with `load_state_dict()`.
- `read_metadata(path) -> dict`: reads the metadata without building the network.

**Errors** — all raise `ValueError` (→ stderr + exit 84 in the executables): missing/unreadable file, file that is not a zip archive (text, single `.npy`, truncated file), wrong `format` or `version`, missing or invalid `config`, missing / wrongly shaped / non-numeric / unexpected arrays.

**Not saved**: the optimizer state (momentum, Adam moments). Retraining a loaded network starts with a fresh optimizer configured from `config`. This keeps files small; the first steps of a new training simply rebuild the moments.

#### `scripts/generator.py` — `my_torch_generator`
Creates blank (untrained) networks from `.conf` files **[GEN1]**.

```
./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]
```

- Writes `<name>_1.nn` … `<name>_<nb>.nn` **in the current directory**, `<name>` being the config file name without extension (`configs/basic_network.conf 3` → `basic_network_1.nn`, `basic_network_2.nn`, `basic_network_3.nn`), and prints each created file name.
- **Seeds**: if the config has `"seed": s`, network *k* uses seed `s + k - 1` (saved in its config). The networks differ from each other, and running the same command again produces exactly the same files. Without a seed, weights are random.
- If the same config name appears twice, numbering continues (`a.conf 1 a.conf 1` → `a_1.nn`, `a_2.nn`) instead of overwriting.
- **All arguments and configurations are validated before any file is written**: an error never leaves half of the networks generated.
- `--help` prints the usage. Odd number of arguments, `nb` not a positive integer, missing or invalid config, or missing NumPy → message on stderr, exit code **84**.
- Generated `.nn` files are git-ignored, except `my_torch_network*.nn` (the pre-trained networks that must be pushed).

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
    Conf["configs/*.conf"] --> Config["mytorch/config.py (build_from_config)"]
    Config --> Engine
    Gen["scripts/generator.py"] --> Config
    Gen --> Ser["mytorch/serialization.py (.nn)"]
    Main --> Ser
    Ser --> Config
    Layers --> Init["mytorch/initializers.py"]
```

# 09 — Guide d'utilisation de `mytorch`

Ce guide montre **comment se servir de la librairie**, avec des exemples prêts à copier. Pour le *pourquoi* de chaque élément, voir les chapitres [02](02-theorie.md) à [05](05-configuration-et-sauvegarde.md).

Tous les exemples de ce guide ont été exécutés (sauf le tracé de courbes, qui demande matplotlib). Ils s'enchaînent : les variables définies dans une section sont réutilisées dans les suivantes.

---

## 0. Prérequis

```bash
cd OurTorch
python3 -m venv .venv            # une seule fois
source .venv/bin/activate        # à chaque nouveau terminal
pip install numpy                # seule dépendance de mytorch
```

Lancer les scripts **depuis la racine du dépôt**, pour que les paquets `mytorch` et `analyzer` soient trouvés :

```bash
python3 mon_script.py            # mon_script.py à la racine
PYTHONPATH=. python3 scripts/x.py  # script rangé dans un sous-dossier
```

```python
import mytorch                                       # Sequential, losses, optimiseurs, config
from mytorch.layers import Linear, ReLU, Dropout     # couches
from mytorch.serialization import save_network, load_network
from analyzer.fen_encoder import read_chess_file, encode_fens, encode_labels   # échecs → tableaux
```

---

## 1. Démarrage rapide

Le cycle complet en une dizaine de lignes : créer un réseau, préparer les données, entraîner, évaluer, prédire, sauvegarder.

```python
import numpy as np
from mytorch import build_from_config
from mytorch.serialization import save_network, load_network
from analyzer.fen_encoder import read_chess_file, encode_fens, encode_labels

# Un petit fichier d'exemple (positions du sujet), au format de CHESSFILE
with open("exemple_boards.txt", "w") as f:
    f.write("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3 Checkmate Black\n"
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 Nothing\n"
            "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1 Nothing\n"
            "rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3 Check White\n"
            "8/8/8/8/8/8/8/k1K5 w - - 0 1 Nothing\n")

model = build_from_config("configs/medium_network.conf")   # 1. réseau vierge (Nothing / Check / Checkmate)

fens, labels = read_chess_file("exemple_boards.txt", require_label=True)
X = encode_fens(fens)                                       # 2. (5, 773)
y = encode_labels(labels, model.classes)                    #    (5, 3), "Checkmate Black" → "Checkmate"

history = model.fit(X, y, epochs=50, batch_size=4, val_split=0.0)   # 3. entraînement
print(model.evaluate(X, y))                                 # 4. (loss, accuracy)
print(model.predict(X))                                     # 5. ['Checkmate', 'Nothing', 'Nothing', 'Check', 'Nothing']

save_network(model, "mon_reseau.nn")                        # 6. sauvegarde
model = load_network("mon_reseau.nn")                       #    rechargement
```

> 5 positions suffisent pour vérifier que tout fonctionne, pas pour apprendre les échecs : un vrai entraînement demande des dizaines de milliers de positions labellisées.

---

## 2. Créer un réseau

Quatre façons, selon le besoin.

### 2.1 Depuis un fichier `.conf` (recommandé)

```python
from mytorch import build_from_config

model = build_from_config("configs/basic_network.conf")
print(model.classes)            # ['Nothing', 'Check']
print(model.training_config)    # {'epochs': 30, 'batch_size': 128, 'val_split': 0.15, 'patience': 5}
print(model.seed)               # 42
```

La config est validée (erreur précise si un champ est faux), la graine est appliquée : deux appels donnent exactement les mêmes poids. Format complet : [05 §1](05-configuration-et-sauvegarde.md#1-les-fichiers-conf-configpy).

### 2.2 Depuis un dictionnaire (tester des variantes sans écrire de fichier)

```python
from mytorch import build_from_config

config = {
    "classes": ["Nothing", "Check"],
    "optimizer": {"type": "sgd", "lr": 0.05, "momentum": 0.9},
    "seed": 1,
    "layers": [
        {"type": "normalization"},
        {"type": "linear", "in": 773, "out": 64},
        {"type": "leakyrelu", "alpha": 0.05},
        {"type": "linear", "in": 64, "out": 2, "init": "xavier"},
    ],
}
model = build_from_config(config)     # les clés absentes prennent leur valeur par défaut
```

### 2.3 En Python, couche par couche ([GEN2])

```python
import numpy as np
from mytorch import Sequential, SoftmaxCrossEntropyLoss, Adam
from mytorch.layers import Normalization, Linear, ReLU, Dropout

np.random.seed(42)                     # sinon les poids changent à chaque exécution
model = Sequential(
    layers=[
        Normalization(773),
        Linear(773, 128, init="he"), ReLU(), Dropout(0.2),
        Linear(128, 2, init="xavier"),            # sortie = logits, pas de Softmax
    ],
    loss=SoftmaxCrossEntropyLoss(),
    optimizer=Adam(lr=0.001, weight_decay=1e-4),
    classes=["Nothing", "Check"],
    training_config={"epochs": 30, "batch_size": 128, "val_split": 0.15, "patience": 5},
    seed=42,
)
```

Différence avec un `.conf` : rien n'est validé à l'avance (une erreur de taille n'apparaît qu'au premier `forward`). `model.get_config()` donne le `.conf` équivalent :

```python
import json
with open("mon_reseau.conf", "w") as f:
    json.dump(model.get_config(), f, indent=2)
```

### 2.4 Plusieurs réseaux d'un coup : `my_torch_generator` ([GEN1])

```bash
make
./my_torch_generator configs/basic_network.conf 3    # basic_network_1.nn … _3.nn, poids différents
```

Détails : [05 §3](05-configuration-et-sauvegarde.md#3-my_torch_generator-scriptsgeneratorpy).

---

## 3. Préparer les données

`mytorch` attend :

| Tableau | Forme | Type | Contenu |
| :--- | :--- | :--- | :--- |
| `X` | `(N, nb_features)` | `float32` | Une ligne par exemple |
| `y` | `(N, nb_classes)` | `float32` | One-hot, colonnes **dans l'ordre de `model.classes`** |

Pour les échecs, `analyzer/fen_encoder.py` fait la conversion ([détails](fen_encoder.MD)) :

| Fonction | Rôle |
| :--- | :--- |
| `read_chess_file(path, require_label)` | Lit un CHESSFILE → `(fens, labels)`. Erreur précise (avec numéro de ligne) si une FEN est invalide, ou si un label manque alors que `require_label=True`. |
| `encode_fens(fens)` | Liste de FEN → `X` de forme `(N, 773)` |
| `encode_labels(labels, classes)` | Labels → `y` one-hot. **Adapte** automatiquement les labels au réseau : `Checkmate Black` devient `Checkmate` pour un réseau intermédiaire, `Check` pour un réseau basique ; `Stalemate` devient `Nothing`. |

Avec d'autres données (pas des échecs), il suffit de construire `X` et `y` soi-même :

```python
import numpy as np
labels_index = np.array([0, 2, 1, 0])                 # classe de chaque exemple
y_autre = np.eye(3, dtype=np.float32)[labels_index]   # one-hot (4, 3)
```

---

## 4. Entraîner

### 4.1 `fit()` et ses options

```python
y = encode_labels(labels, model.classes)   # toujours encoder pour les classes du réseau utilisé
history = model.fit(X, y, epochs=30, batch_size=128, val_split=0.15, patience=5, verbose=True)
```

| Option | Défaut | Effet |
| :--- | :--- | :--- |
| `epochs` | obligatoire | Nombre maximum de passages sur les données |
| `batch_size` | obligatoire | Taille des mini-batchs |
| `val_split` | `0.15` | Part des données mise de côté pour la validation (`0` = pas de validation) |
| `patience` | `5` | Early stopping : arrêt après `patience` époques sans amélioration de `val_loss`. `None` = jamais |
| `validation_data` | `None` | `(X_val, y_val)` : jeu de validation explicite (remplace `val_split`) |
| `verbose` | `False` | Une ligne par époque **sur stderr** |

Pour reprendre exactement les réglages du `.conf` :

```python
history = model.fit(X, y, **model.training_config)
```

### 4.2 Séparer entraînement / validation / test

`val_split` ne crée qu'un jeu de validation. Pour un **score final honnête**, garder en plus un jeu de **test**, jamais vu pendant l'entraînement :

```python
import numpy as np
from mytorch import build_from_config

def split(X, y, val=0.15, test=0.15, seed=0):
    order = np.random.default_rng(seed).permutation(len(X))
    n_test, n_val = int(len(X) * test), int(len(X) * val)
    te, va, tr = order[:n_test], order[n_test:n_test + n_val], order[n_test + n_val:]
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])

# Données factices pour l'exemple (à remplacer par encode_fens / encode_labels)
np.random.seed(0)
X_all = np.random.rand(500, 773).astype(np.float32)
y_all = np.eye(3, dtype=np.float32)[np.random.randint(0, 3, 500)]

(X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split(X_all, y_all)
model = build_from_config("configs/medium_network.conf")
history = model.fit(X_tr, y_tr, epochs=20, batch_size=64, validation_data=(X_va, y_va))
print("score final sur le test :", model.evaluate(X_te, y_te))
```

### 4.3 Lire l'historique

```python
print(history["best_epoch"])            # époque dont les poids ont été gardés
print(history["train_loss"][-1], history["val_loss"][-1])
print(len(history["train_loss"]))       # < epochs si l'early stopping s'est déclenché
```

**Interpréter** :
- `train_loss` et `val_loss` baissent ensemble → l'apprentissage se passe bien.
- `train_loss` baisse, `val_loss` remonte → **surapprentissage** : plus de dropout, plus de `weight_decay`, plus de données, ou un réseau plus petit.
- les deux restent hautes → **sous-apprentissage** : réseau plus grand, plus d'époques, learning rate à ajuster.
- `nan` → learning rate trop grand (ou données non normalisées).

### 4.4 Réentraîner un réseau chargé

```python
import numpy as np
from mytorch.serialization import load_network, save_network

model = load_network("mon_reseau.nn")
if model.seed is not None:
    np.random.seed(model.seed)          # même mélange et mêmes masques de dropout à chaque exécution
model.fit(X_tr, y_tr, epochs=5, batch_size=64, val_split=0.0)
save_network(model, "mon_reseau.nn")
```

La normalisation garde ses statistiques d'origine. L'optimiseur repart de zéro : son état n'est pas sauvegardé.

---

## 5. Évaluer

```python
loss, accuracy = model.evaluate(X_te, y_te)
```

`evaluate` ne modifie ni les poids ni le mode du modèle. Avec des classes rares (les mats), l'accuracy seule est trompeuse : un réseau qui répond toujours `Nothing` peut avoir 90 %. La librairie ne fournit pas encore de métriques par classe ([B3]) ; en attendant, elles se calculent en quelques lignes :

```python
import numpy as np

def confusion_matrix(model, X, y):
    """Lignes = vraie classe, colonnes = classe prédite."""
    true = np.argmax(y, axis=1)
    pred = np.argmax(model.predict_proba(X), axis=1)
    k = len(model.classes)
    matrix = np.zeros((k, k), dtype=int)
    np.add.at(matrix, (true, pred), 1)
    return matrix

def per_class_report(model, X, y):
    m = confusion_matrix(model, X, y)
    for i, name in enumerate(model.classes):
        tp = m[i, i]
        precision = tp / m[:, i].sum() if m[:, i].sum() else 0.0   # parmi les prédits « name », combien sont justes
        recall = tp / m[i, :].sum() if m[i, :].sum() else 0.0      # parmi les vrais « name », combien sont trouvés
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(f"{name:16s} précision {precision:.2f}  rappel {recall:.2f}  F1 {f1:.2f}")

print(confusion_matrix(model, X_te, y_te))
per_class_report(model, X_te, y_te)
```

---

## 6. Prédire

```python
model.predict(X)                 # ['Nothing', 'Check', ...] : un nom de classe par ligne
model.predict_proba(X)           # (N, nb_classes) probabilités, chaque ligne somme à 1
```

Les deux passent le modèle en **mode inférence** (dropout désactivé) et traitent les données par paquets de 1024 (`batch_size=` pour changer).

**Probabilité de la classe choisie** (confiance du réseau) :

```python
import numpy as np
proba = model.predict_proba(X)
confiance = proba.max(axis=1)
```

**MC Dropout** (moyenne et incertitude sur plusieurs passages, [OPT3]) : la couche le permet, la méthode du modèle n'existe pas encore. À la main :

```python
import numpy as np
from mytorch.layers import Dropout

for layer in model.layers:
    if isinstance(layer, Dropout):
        layer.mc_dropout = True          # le dropout reste actif en inférence

samples = np.stack([model.predict_proba(X) for _ in range(30)])   # (30, N, nb_classes)
moyenne, incertitude = samples.mean(axis=0), samples.std(axis=0)

for layer in model.layers:
    if isinstance(layer, Dropout):
        layer.mc_dropout = False         # remettre le comportement normal
```

---

## 7. Sauvegarder et charger

```python
from mytorch.serialization import save_network, load_network, read_metadata

save_network(model, "my_torch_network_medium.nn", metadata={"encoder": "fen-773", "dataset": "v1"})
model = load_network("my_torch_network_medium.nn")
print(read_metadata("my_torch_network_medium.nn"))   # {'encoder': 'fen-773', 'dataset': 'v1'}
```

- Le `.nn` contient l'architecture, les classes, les réglages, les poids **et** les statistiques de normalisation : rien d'autre n'est nécessaire pour prédire.
- La sauvegarde est atomique : en cas d'échec, l'ancien fichier reste intact.
- Toute anomalie au chargement (fichier absent, corrompu, autre format) lève une `ValueError` au message explicite.
- Les réseaux à pousser pour l'autograder doivent s'appeler `my_torch_network*.nn` (les autres `.nn` sont ignorés par git).

---

## 8. Recettes avancées

### 8.1 Comparer plusieurs configurations (recherche d'hyperparamètres)

La librairie n'a pas encore d'outil de recherche ([OPT2]), mais une **grid search** s'écrit avec `build_from_config` sur des dictionnaires :

```python
import copy, itertools
from mytorch import build_from_config
from mytorch.config import load_config

base = load_config("configs/medium_network.conf")
results = []
for lr, rate in itertools.product([0.001, 0.003], [0.1, 0.3]):
    config = copy.deepcopy(base)
    config["optimizer"]["lr"] = lr
    for layer in config["layers"]:
        if layer["type"] == "dropout":
            layer["rate"] = rate
    model = build_from_config(config)
    history = model.fit(X_tr, y_tr, epochs=10, batch_size=64, validation_data=(X_va, y_va))
    best = min(history["val_loss"])
    results.append((best, lr, rate))
    print(f"lr={lr} dropout={rate} → meilleure val_loss {best:.4f}")

print("meilleure combinaison :", min(results))
```

Toujours **choisir sur la validation**, puis mesurer une seule fois sur le test le modèle retenu.

### 8.2 Tracer les courbes d'apprentissage ([B2])

Nécessite `pip install matplotlib` (autorisé par le sujet pour la visualisation) :

```python
import matplotlib.pyplot as plt

plt.plot(history["train_loss"], label="entraînement")
plt.plot(history["val_loss"], label="validation")
plt.axvline(history["best_epoch"] - 1, linestyle="--", label="meilleure époque")
plt.xlabel("époque"); plt.ylabel("loss"); plt.legend()
plt.savefig("courbe.png")
```

Pour comparer plusieurs modèles sur un même graphique, appeler `plt.plot` une fois par `history`.

### 8.3 Boucle d'entraînement manuelle (planning de learning rate)

`fit()` couvre la plupart des besoins. Pour un contrôle total, `train_step()` fait un seul pas d'apprentissage. Exemple avec un learning rate divisé par 2 toutes les 5 époques :

```python
import numpy as np
from mytorch import build_from_config

model = build_from_config("configs/basic_network.conf")
y_basic = np.eye(2, dtype=np.float32)[np.random.randint(0, 2, len(X_tr))]
model.layers[0].adapt(X_tr)                  # fit() le fait d'habitude tout seul
model.train()                                # active le dropout
for epoch in range(15):
    model.optimizer.lr = 0.001 * 0.5 ** (epoch // 5)
    order = np.random.permutation(len(X_tr))
    for start in range(0, len(X_tr), 64):
        idx = order[start:start + 64]
        loss = model.train_step(X_tr[idx], y_basic[idx])
model.eval()                                 # à ne pas oublier avant de prédire
```

Dans une boucle manuelle, c'est à toi de gérer la normalisation, les modes `train()` / `eval()`, la validation et l'early stopping.

### 8.4 Ajouter une couche

Voir [03 — Écrire une nouvelle couche](03-couches.md#écrire-une-nouvelle-couche--exemple-minimal) et la vérifier avec le gradient check ([07](07-tests.md#ajouter-une-nouvelle-couche)).

---

## 9. Erreurs fréquentes

| Message (extrait) | Cause | Solution |
| :--- | :--- | :--- |
| `Normalization expected input of shape (batch, 773), got (N, 789)` | Données encodées pour une autre taille que le réseau | Vérifier `INPUT_SIZE` de l'encodeur et le premier `linear.in` du `.conf` |
| `Linear expected input of shape (batch, 64), got …` | Tailles incohérentes entre deux couches (réseau construit en Python) | Le `in` d'une couche = le `out` de la précédente |
| `Loss shape mismatch: y_pred (N, 3) vs y_true (N, 5)` | `y` encodé pour d'autres classes que celles du réseau | `encode_labels(labels, model.classes)` |
| `Network has 3 outputs but 2 classes` | `classes` ne correspond pas à la dernière couche | Même nombre de classes que de sorties |
| `Label 'Stalemate' cannot be mapped…` / `Unknown label` | Label absent des classes du réseau | Vérifier le fichier de données et `model.classes` |
| `layers[4].in is 32 but the previous layer outputs 16 features.` | Faute dans un `.conf` | Corriger le champ indiqué |
| `config: unknown key(s) ['learning_rate']` | Clé mal nommée dans un `.conf` | Voir les clés autorisées affichées dans le message |
| `Do not end the network with 'softmax'…` | Couche `softmax` + `cross_entropy` | Retirer la couche `softmax` |
| `'x.nn' is not a valid network file` | Fichier corrompu, ou pas un `.nn` | Régénérer le réseau |
| `Linear.backward() called before forward()` | `backward` appelé seul | Toujours `forward` puis `backward` (ou utiliser `train_step`) |
| Loss = `nan` | Learning rate trop grand, données non normalisées | Baisser `lr`, garder la couche `normalization` |
| `No module named 'mytorch'` | Script lancé hors de la racine | Lancer depuis la racine, ou `PYTHONPATH=.` |
| `No module named 'numpy'` | Environnement virtuel non activé | `source .venv/bin/activate` |

---

## 10. Aide-mémoire de l'API

```python
# --- Construction ---
build_from_config(path_ou_dict) -> Sequential
load_config(path) -> dict                       # lu et validé
validate_config(dict) -> dict                   # copie complétée
Sequential(layers, loss, optimizer, classes, training_config=None, seed=None)

# --- Couches (mytorch.layers) ---
Linear(in_features, out_features, init="he")    # init : he, xavier, xavier_uniform, lecun
ReLU()  LeakyReLU(alpha=0.01)  Sigmoid()  Tanh()  Softmax()
Dropout(rate=0.2, mc_dropout=False)
Normalization(features)                         # première couche uniquement

# --- Losses / optimiseurs ---
SoftmaxCrossEntropyLoss()  MSELoss()            # get_loss("cross_entropy" | "mse")
Adam(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0)
SGD(lr=0.01, momentum=0.9, weight_decay=0.0)    # get_optimizer({"type": ..., ...})

# --- Modèle ---
model.fit(X, y, epochs, batch_size, val_split=0.15, patience=5, validation_data=None, verbose=False) -> history
model.evaluate(X, y) -> (loss, accuracy)
model.predict(X) -> [str]
model.predict_proba(X) -> ndarray (N, nb_classes)
model.train_step(X_batch, y_batch) -> float
model.train() / model.eval()
model.get_config() -> dict                      # format .conf
model.classes, model.training_config, model.seed, model.layers, model.optimizer

# --- Fichiers ---
save_network(model, path, metadata=None)
load_network(path) -> Sequential
read_metadata(path) -> dict

# --- Échecs (analyzer.fen_encoder) ---
read_chess_file(path, require_label) -> (fens, labels)
encode_fens(fens) -> ndarray (N, 773)
encode_labels(labels, classes) -> ndarray (N, nb_classes)
```

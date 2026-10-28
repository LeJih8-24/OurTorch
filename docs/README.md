# Documentation OurTorch (français)

Cette documentation explique **chaque élément du code** : ce qu'il fait, pourquoi il est conçu ainsi, comment l'utiliser, et comment le défendre en soutenance.

> Le [README](../README.md) et [ARCHITECTURE.md](../ARCHITECTURE.md) à la racine restent la référence courte (en anglais). Ce dossier est la version détaillée, en français.

## Par où commencer ?

| Si tu veux… | Lis |
| :--- | :--- |
| Comprendre le projet et l'organisation du code en 10 minutes | [01 — Vue d'ensemble](01-vue-d-ensemble.md) |
| Comprendre comment un réseau de neurones apprend (forward, backpropagation, gradients) | [02 — Théorie](02-theorie.md) |
| Comprendre une couche précise (`Linear`, `ReLU`, `Dropout`, `Normalization`…) | [03 — Les couches](03-couches.md) |
| Comprendre l'entraînement : losses, optimiseurs, `Sequential.fit()` | [04 — Entraînement](04-entrainement.md) |
| Écrire un `.conf`, sauvegarder / charger un `.nn`, utiliser `my_torch_generator` | [05 — Configuration et sauvegarde](05-configuration-et-sauvegarde.md) |
| Comprendre `my_torch_analyzer`, la gestion d'erreurs et le `Makefile` | [06 — Analyseur et outils](06-analyseur-et-outils.md) |
| Lancer les tests, comprendre le gradient check, écrire un nouveau test | [07 — Tests](07-tests.md) |
| Préparer la soutenance : questions probables, conformité au sujet, limites connues | [08 — Soutenance](08-soutenance.md) |
| **Utiliser la librairie** : exemples prêts à copier, recettes, erreurs fréquentes, aide-mémoire | [09 — Guide d'utilisation](09-guide-utilisation.md) |
| Comprendre l'encodage des positions FEN | [Encodage FEN](fen_encoder.MD) |

## Carte de la librairie

```
mytorch/
├── initializers.py      → comment on tire les poids au départ           (03)
├── layers/
│   ├── base.py          → le contrat commun à toutes les couches        (03)
│   ├── dense.py         → Linear : Y = XW + b                           (03)
│   ├── activations.py   → ReLU, LeakyReLU, Sigmoid, Tanh, Softmax       (03)
│   ├── regularization.py→ Dropout / MC Dropout                          (03)
│   └── normalization.py → Normalization : standardisation des entrées   (03)
├── losses.py            → MSE, cross-entropy : mesurer l'erreur          (04)
├── optimizers.py        → SGD, Adam : corriger les poids                 (04)
├── model.py             → Sequential : assembler et entraîner            (04)
├── config.py            → fichiers .conf (JSON)                          (05)
└── serialization.py     → fichiers .nn (sauvegarde / chargement)         (05)
```

## Glossaire express

| Terme | Signification dans ce projet |
| :--- | :--- |
| **Batch / mini-batch** | Paquet de positions traitées ensemble. Toutes les matrices sont de forme `(batch, features)` : une ligne = une position d'échecs. |
| **Feature** | Un nombre décrivant la position (ex. « il y a un cavalier blanc en e4 » → 1.0). L'encodeur FEN (`analyzer/fen_encoder.py`) produit 773 features par position ([détails](fen_encoder.MD)). |
| **Logits** | Sorties brutes du réseau, avant softmax. Un logit par classe. |
| **One-hot** | Encodage d'un label : `[0, 1, 0]` pour la 2ᵉ classe. |
| **Forward** | Calcul de la sortie à partir de l'entrée, couche après couche. |
| **Backward / backpropagation** | Calcul des gradients, couche après couche en sens inverse. |
| **Gradient** | Dérivée de la loss par rapport à un nombre (poids ou entrée) : indique dans quel sens le modifier pour réduire l'erreur. |
| **Loss** | Nombre qui mesure l'erreur du réseau sur un batch. On cherche à la minimiser. |
| **Époque (epoch)** | Un passage complet sur tout le jeu d'entraînement. |
| **Overfitting (surapprentissage)** | Le réseau apprend par cœur le jeu d'entraînement mais généralise mal à de nouvelles positions. |
| **Inférence / prédiction** | Utiliser le réseau entraîné, sans modifier ses poids. |

# 03 — Les couches (`mytorch/layers/` et `initializers.py`)

Toutes les couches s'importent depuis `mytorch.layers` :

```python
from mytorch.layers import Layer, Linear, ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, Dropout, Normalization
```

| Couche | Paramètres appris | État sauvegardé | Rôle |
| :--- | :--- | :--- | :--- |
| `Linear` | `W`, `b` | `W`, `b` | Combinaison linéaire $XW + b$ |
| `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `Softmax` | — | — | Non-linéarité |
| `Dropout` | — | — | Régularisation (anti-overfitting) |
| `Normalization` | — | `mean`, `std`, `count` | Standardisation des entrées |

---

## Le contrat commun : `Layer` (`base.py`)

`Layer` est une **classe abstraite** (`ABC`) : on ne peut pas l'instancier directement (`Layer()` lève `TypeError`). Toute couche hérite d'elle et doit implémenter `forward` et `backward`.

| Élément | Signature | Rôle |
| :--- | :--- | :--- |
| `training` | attribut `bool`, `True` par défaut | Mode entraînement (`True`) ou inférence (`False`). Seul `Dropout` change de comportement selon ce mode. Modifié par `Sequential.train()` / `eval()`. |
| `forward` | `forward(inputs) -> ndarray` | **Abstraite.** Calcule la sortie et met en cache ce dont `backward` aura besoin. |
| `backward` | `backward(grad_output) -> ndarray` | **Abstraite.** Reçoit $\frac{\partial L}{\partial Y}$, remplit les gradients des paramètres, renvoie $\frac{\partial L}{\partial X}$. |
| `parameters` | `parameters() -> [(param, grad), ...]` | Paires `(tableau de poids, tableau de gradients)` que l'optimiseur met à jour. `[]` par défaut. |
| `get_config` | `get_config() -> dict` | Description de la couche **au format d'une entrée de `.conf`** (ex. `{"type": "relu"}`). Sert à sauvegarder l'architecture. |
| `state_keys` | attribut de classe, tuple de noms | Noms des attributs tableaux à sauvegarder dans un `.nn`. Vide par défaut. |
| `state_dict` | `state_dict() -> {nom: tableau}` | Les tableaux listés dans `state_keys`. |
| `load_state_dict` | `load_state_dict(state)` | Recopie des tableaux sauvegardés **en place**. Vérifie que les clés et les formes correspondent, sinon `ValueError`. |

### Conventions (à respecter pour toute nouvelle couche)

1. **Tout est 2D** : `(batch, features)`, une ligne par exemple.
2. `forward()` met en cache ce dont `backward()` a besoin. Appeler `backward()` avant `forward()` lève `RuntimeError`.
3. Les gradients des paramètres sont **écrasés** à chaque `backward()` (pas accumulés).
4. Les gradients sont alloués une fois puis écrits **en place** : `self.dW[...] = ...`, **jamais** `self.dW = ...`.
   *Pourquoi ?* L'optimiseur garde des références vers ces tableaux (et indexe son état interne, comme le momentum, par `id(tableau)`). Réassigner l'attribut créerait un nouveau tableau que l'optimiseur ne verrait pas.
5. `get_config()` renvoie exactement le format d'une couche dans un `.conf`, pour que `build_from_config(model.get_config())` reconstruise le même réseau.

### Écrire une nouvelle couche — exemple minimal

```python
import numpy as np
from mytorch.layers.base import Layer

class Scale(Layer):
    """Y = 2 * X (exemple pédagogique)."""

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return 2.0 * inputs

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return 2.0 * grad_output          # dL/dX = dL/dY * dY/dX = G * 2

    def get_config(self):
        return {"type": "scale"}
```

Pour l'utiliser dans un `.conf`, il faudrait aussi l'ajouter à `LAYER_SPECS` et `_BUILDERS` dans `config.py` ([05](05-configuration-et-sauvegarde.md)), puis la valider avec `check_layer_gradients` ([07](07-tests.md)).

---

## Initialisation des poids (`initializers.py`)

Toutes les fonctions prennent une forme `(n_in, n_out)` et renvoient un tableau `float32`. Une forme invalide (pas 2D, dimension ≤ 0) lève `ValueError`.

| Fonction | Distribution | Usage typique |
| :--- | :--- | :--- |
| `he_normal(shape)` | $\mathcal{N}(0, \sigma^2)$, $\sigma = \sqrt{2/n_{in}}$ | Couches suivies de ReLU / LeakyReLU |
| `xavier_normal(shape)` | $\sigma = \sqrt{2/(n_{in}+n_{out})}$ | Tanh, Sigmoid, couche de sortie |
| `xavier_uniform(shape)` | $\mathcal{U}(-a, a)$, $a = \sqrt{6/(n_{in}+n_{out})}$ | Même usage, distribution uniforme |
| `lecun_normal(shape)` | $\sigma = \sqrt{1/n_{in}}$ | SELU |

**`get_initializer(name)`** renvoie la fonction correspondant à un nom (insensible à la casse et aux espaces) ; nom inconnu → `ValueError`. Noms acceptés :

| Nom(s) | Fonction |
| :--- | :--- |
| `he`, `kaiming`, `he_normal`, `kaiming_normal` | `he_normal` |
| `xavier`, `glorot`, `xavier_normal`, `glorot_normal` | `xavier_normal` |
| `xavier_uniform`, `glorot_uniform` | `xavier_uniform` |
| `lecun`, `lecun_normal` | `lecun_normal` |

Le tirage utilise le générateur aléatoire global de NumPy : `np.random.seed(s)` avant de créer le réseau rend les poids reproductibles (fait automatiquement par `build_from_config` quand le `.conf` contient une graine).

Justification théorique : [02 — Théorie §7](02-theorie.md#7-initialisation-des-poids).

---

## `Linear` (`dense.py`)

```python
Linear(in_features: int, out_features: int, init: str = "he")
```

Couche entièrement connectée : chaque sortie est une combinaison de toutes les entrées.

| Attribut | Forme | Contenu |
| :--- | :--- | :--- |
| `W` | `(in_features, out_features)` | Poids, initialisés par `get_initializer(init)`, `float32` |
| `b` | `(1, out_features)` | Biais, initialisés à 0 |
| `dW`, `db` | mêmes formes | Gradients, remplis par `backward()` |
| `in_features`, `out_features`, `init` | — | Configuration |

| Méthode | Calcul |
| :--- | :--- |
| `forward(X)` | Vérifie que `X` est `(N, in_features)`, mémorise `X`, renvoie $XW + b$ |
| `backward(G)` | $dW = X^\top G$, $db = \sum_{\text{lignes}} G$, renvoie $G W^\top$ |
| `parameters()` | `[(W, dW), (b, db)]` |
| `get_config()` | `{"type": "linear", "in": …, "out": …, "init": …}` |
| `state_keys` | `("W", "b")` |

Erreurs : dimensions ≤ 0 ou initialiseur inconnu → `ValueError` à la construction ; mauvaise largeur d'entrée → `ValueError` ; `backward` avant `forward` → `RuntimeError`.

**Nombre de paramètres** : $n_{in} \times n_{out} + n_{out}$. Pour `Linear(789, 256)` : 202 240 paramètres.

```python
layer = Linear(3, 2, init="xavier")
Y = layer.forward(np.ones((4, 3), dtype=np.float32))   # (4, 2)
dX = layer.backward(np.ones((4, 2), dtype=np.float32)) # (4, 3), remplit layer.dW et layer.db
```

---

## Activations (`activations.py`)

Toutes héritent d'une petite classe `Activation` qui gère le cache (`_cache`) et l'erreur `backward avant forward`. Elles n'ont **aucun paramètre** (`parameters() == []`) et ne changent pas la forme des données.

| Classe | Forward | Backward ($G$ = gradient reçu) | Ce qui est mis en cache | Config |
| :--- | :--- | :--- | :--- | :--- |
| `ReLU()` | $\max(0, x)$ | $G \cdot [x > 0]$ | l'entrée $x$ | `{"type": "relu"}` |
| `LeakyReLU(alpha=0.01)` | $x$ si $x>0$, sinon $\alpha x$ | $G \cdot (1$ ou $\alpha)$ | l'entrée $x$ | `{"type": "leakyrelu", "alpha": 0.01}` |
| `Sigmoid()` | $\sigma(x) = \frac{1}{1+e^{-x}}$ | $G \cdot \sigma(1-\sigma)$ | la sortie $\sigma$ | `{"type": "sigmoid"}` |
| `Tanh()` | $\tanh(x)$ | $G \cdot (1 - \tanh^2)$ | la sortie | `{"type": "tanh"}` |
| `Softmax()` | $\frac{e^{x_i - \max x}}{\sum_j e^{x_j - \max x}}$ par ligne | $s \odot (G - \langle G, s\rangle)$ | la sortie $s$ | `{"type": "softmax"}` |

**Quand utiliser laquelle ?**
- **ReLU** : le choix par défaut pour les couches cachées. Simple, rapide, ne sature pas pour $x > 0$ (les gradients ne disparaissent pas).
- **LeakyReLU** : comme ReLU mais avec une petite pente pour $x < 0$. Évite les « neurones morts » (un neurone ReLU dont l'entrée est toujours négative ne reçoit plus jamais de gradient).
- **Sigmoid / Tanh** : historiques. Elles **saturent** (dérivée ≈ 0 pour les grandes valeurs), ce qui ralentit l'apprentissage des réseaux profonds. Utiles pour comparer dans les benchmarks.
- **Softmax** : transforme des logits en probabilités. **Ne pas la mettre en dernière couche avec la cross-entropy** : la loss l'applique déjà (la validation des `.conf` refuse cette combinaison). Voir [02 §6](02-theorie.md#le-cas-élégant-softmax--cross-entropy).

**Détails de stabilité** :
- `Sigmoid` limite l'entrée à $[-50, 50]$ avant l'exponentielle, pour éviter un dépassement de capacité ($e^{1000}$).
- `Softmax` soustrait le maximum de chaque ligne avant l'exponentielle.

**Le backward de `Softmax`** : chaque sortie dépend de toutes les entrées de la ligne, donc la dérivée est une matrice (jacobienne) $J_{ij} = s_i(\delta_{ij} - s_j)$. Plutôt que de la construire, on calcule directement son produit avec $G$ : $s \odot (G - \sum_j G_j s_j)$.

---

## `Dropout` (`regularization.py`)

```python
Dropout(rate: float = 0.2, mc_dropout: bool = False)
```

**Idée** : à chaque passage en entraînement, chaque neurone est éteint (mis à 0) avec la probabilité `rate`. Le réseau ne peut pas compter sur un neurone particulier et doit répartir l'information : il généralise mieux. C'est une mesure anti-overfitting **[OPT1]**.

**Dropout « inversé »** : les neurones survivants sont multipliés par $\frac{1}{1 - \text{rate}}$. Ainsi, **la valeur moyenne ne change pas**, et en prédiction il suffit de ne rien faire (pas de correction).

| Mode | Comportement de `forward(X)` |
| :--- | :--- |
| Entraînement (`training=True`) | Tire un masque aléatoire, renvoie `X * masque` (valeurs 0 ou `1/(1-rate)`) |
| Inférence (`training=False`) | Renvoie `X` tel quel (identité) |
| Inférence avec `mc_dropout=True` | Reste actif (masque aléatoire) |

- `backward(G)` multiplie `G` par **le même masque** : un neurone éteint ne reçoit pas de gradient.
- `active` (propriété) : `True` si le dropout s'applique en ce moment.
- `rate` doit être dans $[0, 1)$, sinon `ValueError`. `rate=0` → identité.
- Masque tiré avec `np.random` : reproductible avec `np.random.seed()`.
- Config : `{"type": "dropout", "rate": 0.2, "mc_dropout": false}`.

**MC Dropout ([OPT3])** : avec `mc_dropout=True`, on peut prédire $T$ fois la même position avec des masques différents. La moyenne des $T$ prédictions est souvent plus fiable, et leur dispersion mesure l'**incertitude** du réseau. La couche le permet ; une méthode `Sequential` dédiée (`predict_mc`) reste à écrire.

---

## `Normalization` (`normalization.py`)

```python
Normalization(features: int)
```

Standardise chaque feature : $Y = \frac{X - \mu}{\sigma}$, pour que toutes aient une moyenne 0 et un écart-type 1. Justification : [02 — Théorie §8](02-theorie.md#8-normalisation-des-entrées).

| Attribut | Forme | Contenu |
| :--- | :--- | :--- |
| `mean` ($\mu$) | `(1, features)` | Moyenne de chaque feature, `0` avant `adapt` |
| `std` ($\sigma$) | `(1, features)` | Écart-type de chaque feature, `1` avant `adapt` |
| `count` | `(1,)` | Nombre d'exemples utilisés par `adapt` (0 = pas encore calculé) |
| `adapted` (propriété) | — | `count > 0` |

| Méthode | Rôle |
| :--- | :--- |
| `adapt(X)` | Calcule $\mu$ et $\sigma$ sur `X` (le jeu d'entraînement). Les features d'écart-type < $10^{-6}$ gardent $\sigma = 1$. |
| `forward(X)` | $(X - \mu) / \sigma$. Avant `adapt` : identité. |
| `backward(G)` | $G / \sigma$ |
| `get_config()` | `{"type": "normalization", "features": N}` |
| `state_keys` | `("mean", "std", "count")` |

**Points importants** :
- $\mu$ et $\sigma$ **ne sont pas appris** par l'optimiseur (`parameters()` est vide) : ils sont calculés une fois, puis **sauvegardés dans le `.nn`**. La prédiction applique ainsi exactement la même transformation que l'entraînement.
- **Features constantes** (ex. une case qui est vide dans toutes les positions d'entraînement) : leur écart-type est 0. On garde $\sigma = 1$ pour ne jamais diviser par 0 : la feature est seulement centrée.
- **Uniquement en première couche** : imposé par `Sequential` et par la validation des `.conf`, car les statistiques sont calculées sur les entrées brutes.
- **Calcul automatique** : `Sequential.fit()` appelle `adapt()` sur la **partie entraînement seulement** (jamais sur la validation, ce qui fausserait l'évaluation), et **une seule fois** : réentraîner un réseau chargé conserve ses statistiques.
- Pourquoi `count` est sauvegardé : pour distinguer un réseau vierge (statistiques à calculer) d'un réseau déjà adapté, après un aller-retour dans un fichier `.nn`.

```python
norm = Normalization(2)
norm.adapt(np.array([[0.0, 10.0], [2.0, 30.0]]))
norm.mean   # [[ 1. 20.]]
norm.std    # [[ 1. 10.]]
norm.forward(np.array([[2.0, 30.0]]))   # [[1. 1.]]
```

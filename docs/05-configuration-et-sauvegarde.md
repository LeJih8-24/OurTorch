# 05 — Configuration (`.conf`), sauvegarde (`.nn`) et générateur

Deux formats de fichiers, deux rôles :

| | `.conf` | `.nn` |
| :--- | :--- | :--- |
| Contient | L'**architecture** et les réglages (pas de poids) | **Tout** le réseau : architecture + poids + statistiques de normalisation |
| Format | JSON, écrit à la main | Archive `.npz` (NumPy), écrite par le programme |
| Créé par | Un humain | `my_torch_generator`, `my_torch_analyzer --train` |
| Lu par | `build_from_config`, `my_torch_generator` | `load_network`, `my_torch_analyzer` |
| Module | `mytorch/config.py` | `mytorch/serialization.py` |

---

## 1. Les fichiers `.conf` (`config.py`)

### Exemple commenté (`configs/medium_network.conf`)

```jsonc
{
  "classes": ["Nothing", "Check", "Checkmate"],       // sorties, dans l'ordre des neurones
  "loss": "cross_entropy",                            // ou "mse"
  "optimizer": {"type": "adam", "lr": 0.001, "weight_decay": 0.0001},
  "training": {"epochs": 40, "batch_size": 128, "val_split": 0.15, "patience": 5},
  "seed": 42,                                         // graine : poids reproductibles
  "layers": [
    {"type": "normalization"},                        // taille déduite : 789
    {"type": "linear", "in": 789, "out": 256, "init": "he"},
    {"type": "relu"},
    {"type": "dropout", "rate": 0.2},
    {"type": "linear", "in": 256, "out": 128, "init": "he"},
    {"type": "relu"},
    {"type": "dropout", "rate": 0.2},
    {"type": "linear", "in": 128, "out": 3, "init": "xavier"}
  ]
}
```

(Les commentaires `//` ne sont là que pour l'explication : le JSON réel n'en accepte pas.)

**Pourquoi JSON ?** Lisible et modifiable à la main, versionnable avec git, lu par la bibliothèque standard Python (aucune dépendance), et identique à la sortie de `get_config()`.

### Clés de premier niveau

| Clé | Obligatoire | Défaut | Contenu |
| :--- | :--- | :--- | :--- |
| `classes` | oui | — | Liste d'au moins 2 noms, non vides, sans doublon |
| `layers` | oui | — | Liste de couches (tableau suivant) |
| `loss` | non | `"cross_entropy"` | `"cross_entropy"` ou `"mse"` |
| `optimizer` | non | `{"type": "adam"}` | `type` (`"adam"` / `"sgd"`) + paramètres : `lr`, `weight_decay`, `momentum` (SGD), `beta1`, `beta2`, `eps` (Adam) |
| `training` | non | `{"epochs": 30, "batch_size": 128, "val_split": 0.15, "patience": 5}` | Réglages de `fit()`. `patience: null` désactive l'early stopping |
| `seed` | non | `null` | Entier ≥ 0, ou `null` pour des poids aléatoires différents à chaque fois |

### Types de couches

| `type` | Paramètres (défaut) |
| :--- | :--- |
| `normalization` | `features` (déduit du `in` de la première `linear`). Uniquement en premier. |
| `linear` | `in`, `out` (obligatoires), `init` (`"he"`) |
| `relu`, `sigmoid`, `tanh`, `softmax` | aucun |
| `leakyrelu` | `alpha` (`0.01`) |
| `dropout` | `rate` (`0.2`), `mc_dropout` (`false`) |

### Validation : pourquoi si stricte ?

Une faute de frappe ne doit jamais produire silencieusement un autre réseau que celui voulu. `validate_config()` refuse :
- les **clés inconnues** (ex. `"learning_rate"` au lieu de `optimizer.lr`, `"dropout_rate"` au lieu de `rate`) ;
- les **mauvais types ou valeurs** (`"out": -16`, `"rate": 1.0`, `"epochs": 0`, `true` à la place d'un entier) ;
- les **tailles incohérentes** : le `in` de chaque `linear` doit égaler le `out` de la précédente, et la dernière sortie doit égaler le nombre de classes ;
- `normalization` ailleurs qu'en premier, un réseau sans `linear` ;
- `softmax` en dernière couche avec `cross_entropy` (softmax appliquée deux fois).

Chaque erreur est une `ValueError` qui **nomme le champ fautif** :
```
Error: Invalid configuration file 'net.conf': layers[4].in is 32 but the previous layer outputs 16 features.
```

### Fonctions

| Fonction | Rôle |
| :--- | :--- |
| `validate_config(dict) -> dict` | Vérifie et renvoie une **copie complétée** (valeurs par défaut ajoutées, noms normalisés). Ne modifie pas le dictionnaire reçu. |
| `load_config(path) -> dict` | Lit le fichier JSON puis le valide. Fichier illisible ou JSON invalide → `ValueError` avec le numéro de ligne. |
| `build_from_config(path_ou_dict) -> Sequential` | Valide, appelle `np.random.seed(seed)` si une graine est donnée, crée couches / loss / optimiseur, renvoie un `Sequential` neuf. |

```python
from mytorch import build_from_config

model = build_from_config("configs/medium_network.conf")
history = model.fit(X, y, **model.training_config)    # réglages du fichier
```

`build_from_config(model.get_config())` reconstruit la même architecture (et les mêmes poids initiaux si une graine est fixée).

### Les trois configurations fournies (`configs/`)

| Fichier | Classes | Couches cachées | Dropout | Époques max |
| :--- | :--- | :--- | :--- | :--- |
| `basic_network.conf` | 2 | 128 | 0.2 | 30 |
| `medium_network.conf` | 3 | 256 → 128 | 0.2 | 40 |
| `full_network.conf` | 5 | 512 → 256 | 0.3 | 50 (patience 7) |

Toutes : normalisation en entrée, Adam (`lr=0.001`, `weight_decay=1e-4`), He pour les couches cachées, Xavier en sortie, graine 42. Plus il y a de classes, plus le réseau est large (et plus il est régularisé).

⚠️ **Taille d'entrée 789** : c'est la taille prévue par l'encodeur FEN (issue #2). Si l'encodeur change de taille, il faut modifier le premier `linear.in` de chaque fichier.

---

## 2. Les fichiers `.nn` (`serialization.py`)

### Contenu

Un `.nn` est une **archive `.npz` compressée** (un zip de tableaux NumPy) :

| Entrée | Contenu |
| :--- | :--- |
| `header` | JSON : `{"format": "mytorch-nn", "version": 1, "metadata": {...}}` |
| `config` | JSON : `model.get_config()` (même structure qu'un `.conf`) |
| `layer{i}.{clé}` | Un tableau par élément d'état : `layer0.mean`, `layer0.std`, `layer0.count`, `layer1.W`, `layer1.b`, … |

Pour regarder dedans : `unzip -l basic_network_1.nn`.

### Pourquoi ce format plutôt que `pickle` (proposé dans l'issue) ?

| Critère | `.npz` + JSON | `pickle` |
| :--- | :--- | :--- |
| **Sécurité** | Chargé avec `allow_pickle=False` : un fichier ne peut **jamais exécuter de code** | Ouvrir un pickle piégé exécute du code arbitraire |
| **Robustesse** | Indépendant des noms de classes Python : renommer ou déplacer du code ne casse pas les anciens fichiers | Dépend de la structure exacte du code |
| **Lisibilité** | Config en JSON, tableaux listables | Binaire opaque |
| **Taille** | Compressé, poids en `float32` (~1,9 Mo pour `full_network`) | Similaire |
| **Évolution** | Champ `version` : un fichier d'une autre version est refusé avec un message clair | Rien |

### Fonctions

| Fonction | Rôle |
| :--- | :--- |
| `save_network(model, path, metadata=None)` | Sauvegarde le réseau. `metadata` : dictionnaire JSON libre stocké dans l'en-tête (prévu : version de l'encodeur FEN). |
| `load_network(path) -> Sequential` | Reconstruit l'architecture avec `build_from_config(config)`, puis charge chaque tableau avec `load_state_dict()`. |
| `read_metadata(path) -> dict` | Lit les métadonnées sans construire le réseau. |

```python
from mytorch.serialization import save_network, load_network

save_network(model, "my_torch_network_medium.nn", metadata={"encoder": "fen-v1"})
model = load_network("my_torch_network_medium.nn")
print(model.predict(X[:5]))
```

### Écriture atomique

`save_network` écrit d'abord dans un fichier temporaire du même dossier, puis le **renomme** à la place du fichier final (opération atomique). Si l'écriture échoue en cours de route (disque plein, interruption), l'ancien fichier reste **intact**. Important pour `--train` sans `--save`, qui écrase le `LOADFILE`.

Le fichier final reçoit les permissions habituelles (`666` moins le umask, soit `rw-rw-r--` en général).

### Erreurs détectées au chargement

Toutes lèvent `ValueError` (→ message sur stderr + code 84 dans les exécutables) :
- fichier absent ou illisible ;
- fichier qui n'est pas une archive zip (texte, `.npy` seul, fichier tronqué) ;
- `format` différent de `mytorch-nn`, ou `version` différente de 1 ;
- `config` absente, JSON invalide, ou configuration invalide ;
- tableau manquant, de mauvaise forme, non numérique, ou tableau inattendu en trop ;
- objet Python sérialisé (pickle) dans un tableau.

### Ce qui n'est pas sauvegardé

L'**état de l'optimiseur** (vitesse du momentum, moments d'Adam). Réentraîner un réseau chargé repart avec un optimiseur neuf, configuré selon `config`. Cela garde des fichiers légers ; les premiers pas du nouvel entraînement reconstruisent ces moyennes.

---

## 3. `my_torch_generator` (`scripts/generator.py`)

Crée des réseaux **vierges** depuis des `.conf` **[GEN1]**.

```bash
./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]
```

```bash
$ ./my_torch_generator configs/basic_network.conf 3 configs/full_network.conf 1
basic_network_1.nn
basic_network_2.nn
basic_network_3.nn
full_network_1.nn
```

| Règle | Détail |
| :--- | :--- |
| Noms de sortie | `<nom du .conf sans extension>_<k>.nn`, dans le **dossier courant** ; chaque nom créé est affiché |
| Graines | Avec `"seed": s`, le réseau *k* utilise la graine `s + k - 1` (enregistrée dans son `.nn`) : les réseaux sont **différents**, et relancer la même commande redonne **exactement** les mêmes fichiers. Sans graine : aléatoire. |
| Nom répété | `a.conf 1 a.conf 1` → `a_1.nn`, `a_2.nn` (la numérotation continue, rien n'est écrasé) |
| Validation | **Tous** les arguments et **toutes** les configs sont vérifiés **avant** d'écrire le moindre fichier : une erreur ne laisse jamais une génération à moitié faite |
| `--help` / `-h` | Affiche l'usage, code 0 |
| Erreurs (code 84, message sur stderr) | Nombre d'arguments impair, `nb` qui n'est pas un entier > 0, config absente ou invalide, NumPy non installé |

**Fonctionnement interne** :
- `parse_args(argv)` : vérifie les paires `config nb` et renvoie `[(chemin, nombre), ...]`.
- `output_names(jobs)` : calcule les noms de fichiers en continuant la numérotation des noms répétés.
- `main(argv)` : charge et valide toutes les configs, puis pour chaque réseau fixe la graine, appelle `build_from_config` puis `save_network`.
- Les imports de `mytorch` sont protégés : si NumPy manque, le programme affiche `Error: Cannot load the mytorch library: …` et sort avec 84 au lieu de planter.

Les `.nn` générés sont ignorés par git (`.gitignore`), **sauf** `my_torch_network*.nn` : ce sont les réseaux pré-entraînés que le sujet impose de pousser.

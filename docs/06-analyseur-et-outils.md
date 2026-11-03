# 06 — L'analyseur (`analyzer/`), les erreurs et le `Makefile`

## `my_torch_analyzer` : ce qu'il doit faire

```
./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE
```

| Mode | Entrée (`CHESSFILE`) | Sortie |
| :--- | :--- | :--- |
| `--predict` | Une position FEN par ligne, éventuellement suivie du label attendu | **Une prédiction par ligne sur stdout, dans l'ordre**, rien d'autre (l'autograder compare ligne à ligne) |
| `--train` | Une position FEN par ligne, suivie du label attendu | Entraîne le réseau de `LOADFILE` et le sauvegarde dans `SAVEFILE` si `--save` est donné, sinon **écrase** `LOADFILE`. Affichage libre. |

## État actuel

| Fichier | Rôle | État |
| :--- | :--- | :--- |
| `analyzer/cli.py` | Lecture et validation des arguments | ✅ Fait (#7) |
| `analyzer/utils.py` | `exit_error()` : message sur stderr + code 84 | ✅ Fait (#9) |
| `analyzer/main.py` | Point d'entrée : arguments → pipeline, erreurs → code 84 | ✅ Fait (#1) |
| `analyzer/fen_encoder.py` | FEN → vecteur de 773 nombres, labels → one-hot ([détails](fen_encoder.MD)) | ✅ Fait (#2) |
| `analyzer/pipeline.py` | Modes `--train` et `--predict` | ✅ Fait (#1) |

## `analyzer/utils.py`

```python
exit_error(message: str) -> None
```

Écrit `Error: <message>` suivi d'un retour à la ligne sur **stderr**, puis quitte avec le code **84**. Toutes les erreurs du projet passent par là, ce qui garantit la règle du sujet (« tout message d'erreur sur stderr, code 84 »). Utilisée par `cli.py` et `my_torch_generator`.

## `analyzer/cli.py`

### `AnalyzerArgs`

Une `@dataclass` contenant les arguments validés :

| Champ | Type | Contenu |
| :--- | :--- | :--- |
| `mode` | `str` | `"train"` ou `"predict"` |
| `loadfile` | `str` | Chemin du réseau |
| `chessfile` | `str` | Chemin des positions |
| `savefile` | `Optional[str]` | Chemin de sauvegarde (`--train` uniquement), sinon `None` |

### `parse_args(argv=None) -> AnalyzerArgs`

Pourquoi ne pas utiliser `argparse` ? Parce qu'en cas d'erreur, `argparse` quitte avec le code **2**, alors que le sujet impose **84**. Le parsing est donc écrit à la main.

Règles, dans l'ordre :

1. `-h` ou `--help` **n'importe où** → affiche `HELP_TEXT` sur stdout, code **0**.
2. Les arguments sont lus un par un :
   - `--train` / `--predict` : un seul mode autorisé (deux modes → erreur) ;
   - `--save` : doit être suivi d'un nom de fichier qui ne commence pas par `-`, une seule fois ;
   - toute autre option commençant par `-` → `Unknown option` ;
   - le reste est un argument positionnel.
3. Aucun mode → erreur.
4. `--save` avec `--predict` → erreur.
5. Exactement **2** arguments positionnels (`LOADFILE`, `CHESSFILE`), sinon erreur.
6. Les deux fichiers doivent **exister**, être des **fichiers réguliers**, être **lisibles** et **non vides**.

Chaque erreur passe par `exit_error()` (stderr + code 84). L'ordre des options est libre : `--save out.nn --train a.nn b.txt` est accepté.

`cli.py` vérifie seulement que les fichiers existent et sont lisibles. Le **contenu** (réseau corrompu, FEN invalide, label manquant) est vérifié ensuite par le pipeline.

## `analyzer/main.py` — le point d'entrée

```python
args = parse_args(argv)          # erreurs d'arguments → code 84 (cli.py)
try:
    run(args)                    # pipeline.py
except (ValueError, OSError):    # réseau corrompu, FEN invalide, label manquant, SAVEFILE impossible à écrire…
    exit_error(message)          # stderr + code 84
except MemoryError:
    exit_error("Not enough memory …")
return 0
```

- Toutes les erreurs de la librairie et de l'encodeur sont des `ValueError` aux messages précis : `main.py` n'a qu'un seul endroit où les transformer en code 84.
- L'import du pipeline est protégé : si NumPy n'est pas installé, le programme affiche `Error: Cannot load the mytorch library: …` et sort avec 84 au lieu de planter.

## `analyzer/pipeline.py` — les deux modes

### `--predict`

```
./my_torch_analyzer --predict LOADFILE CHESSFILE
```

1. **Chargement** du réseau (`load_network`) et **vérification** qu'il attend bien les 773 features de l'encodeur (sinon erreur claire : `Network 'x.nn' expects 789 input features, but the FEN encoder produces 773.`).
2. **Lecture** de `CHESSFILE` (`read_chess_file(require_label=False)`) : les lignes vides sont ignorées, un label après la FEN est **accepté et ignoré**, une FEN invalide arrête tout avec son numéro de ligne (`chess.txt:12: …`).
3. **Prédiction par paquets de 1024 positions** (`PREDICT_CHUNK`) : on n'encode jamais tout le fichier d'un coup, la mémoire reste faible même avec des centaines de milliers de positions.
4. **Sortie** : une prédiction par ligne sur **stdout**, dans l'ordre du fichier, **et rien d'autre** (ni sur stdout, ni sur stderr). Les prédictions sont les classes du réseau chargé : `Check` / `Nothing` pour un réseau basique, `Checkmate Black` / `Check White` / `Nothing`… pour un réseau complet.

```
$ ./my_torch_analyzer --predict my_torch_network_full.nn chessboards.txt
Checkmate Black
Nothing
Check White
```

### `--train`

```
./my_torch_analyzer --train [--save SAVEFILE] LOADFILE CHESSFILE
```

1. **Chargement** et vérification de la taille d'entrée, comme en prédiction.
2. **Lecture** avec `require_label=True` : chaque ligne **doit** avoir un label connu, sinon code 84 (`Missing expected output after FEN …` / `Unknown label 'Draw' …`).
3. **Encodage** : `encode_fens` pour les positions, `encode_labels` pour les labels, **adaptés aux classes du réseau** : sur un réseau intermédiaire, `Checkmate White` devient `Checkmate` ; sur un réseau basique, `Check`. On peut donc entraîner les trois niveaux de réseau avec les mêmes fichiers du dataset.
4. **Entraînement** : `np.random.seed(model.seed)` puis `model.fit(X, y, **model.training_config)`. Les réglages (époques, batch, validation, patience) sont ceux du `.conf` dont vient le réseau, et la graine rend l'entraînement **reproductible**. Une ligne par époque est affichée sur **stderr**.
5. **Sauvegarde** dans `SAVEFILE` si `--save` est donné, sinon **à la place de `LOADFILE`**. L'écriture est atomique : si elle échoue (dossier inexistant…), `LOADFILE` reste intact. Les métadonnées du réseau sont conservées et complétées (`input_size`, `trained_on` = nombre de positions).
6. **Résumé** sur stdout (le sujet n'impose rien en mode entraînement) :

```
$ ./my_torch_analyzer --train --save trained.nn medium_network_1.nn train.txt
Trained on 1100 positions (8 epochs) - best epoch 3: val_loss 1.2276, val_acc 0.3879
Network saved to trained.nn
```

**Mémoire** : en entraînement, tout `CHESSFILE` est encodé d'un coup (`fit` a besoin de toutes les données pour mélanger et découper). Chaque position coûte 773 × 4 octets ≈ 3 Ko : 100 000 positions ≈ 300 Mo, le dataset complet (723 000 positions) ≈ 2,2 Go. Pour de très gros entraînements, passer par un script qui utilise directement la librairie.

### Fonctions utiles hors de l'exécutable

| Fonction | Rôle |
| :--- | :--- |
| `load_compatible_network(path)` | `load_network` + vérification de la taille d'entrée |
| `network_input_size(model)` | Nombre de features attendu par la première couche (`Normalization` ou `Linear`) |
| `predict(model, fens, out)` | Écrit les prédictions par paquets sur un flux |
| `run_predict(args)` / `run_train(args)` / `run(args)` | Les deux modes, à partir d'un `AnalyzerArgs` |

### Exemple complet

```bash
make
./my_torch_generator configs/medium_network.conf 1                       # réseau vierge
./my_torch_analyzer --train --save trained.nn medium_network_1.nn train.txt
./my_torch_analyzer --predict trained.nn positions.txt > predictions.txt
```

## Le `Makefile`

Python n'est pas compilé : le `Makefile` copie les points d'entrée à la racine et les rend exécutables. La ligne `#!/usr/bin/env python3` en tête de chaque fichier indique au système quel interpréteur utiliser.

| Règle | Action |
| :--- | :--- |
| `make` / `make all` | Copie `analyzer/main.py` → `./my_torch_analyzer` et `scripts/generator.py` → `./my_torch_generator`, puis `chmod +x` |
| `make clean` | Supprime `__pycache__`, `.pytest_cache`, `*.pyc` |
| `make fclean` | `clean` + supprime les deux exécutables |
| `make re` | `fclean` puis `all` |

**Pourquoi les imports fonctionnent** : quand on lance `./my_torch_generator`, Python ajoute automatiquement le dossier du script (la racine du dépôt) au chemin de recherche des modules. Les paquets `mytorch/` et `analyzer/`, situés à la racine, sont donc trouvés, quel que soit le dossier courant.

**NumPy** : les exécutables utilisent le `python3` du système. NumPy doit être installé pour cet interpréteur (`pip install numpy`), ou il faut activer l'environnement virtuel (`source .venv/bin/activate`). À vérifier sur l'environnement de correction (« the dump »).

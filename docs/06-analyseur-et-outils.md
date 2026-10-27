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
| `analyzer/main.py` | Point d'entrée | ⚠️ Valide les arguments puis renvoie 0 : n'entraîne ni ne prédit encore |
| `analyzer/fen_encoder.py` | FEN → vecteur de 789 nombres, labels → one-hot | ⏳ Issue #2 |
| `analyzer/pipeline.py` | Branchement `--train` / `--predict` | ⏳ Issue #1 |

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

**Remarque pour #1** : `cli.py` vérifie seulement que les fichiers existent. Le contenu (réseau corrompu, FEN invalide) sera vérifié par `load_network()` (qui lève déjà `ValueError`) et par l'encodeur FEN ; le pipeline devra transformer ces `ValueError` en `exit_error`.

## `analyzer/main.py` — ce qu'il reste à brancher (issue #1)

Esquisse du futur pipeline, avec les briques **déjà disponibles** :

```python
import numpy as np
from analyzer.cli import parse_args
from analyzer.utils import exit_error
from mytorch.serialization import load_network, save_network

args = parse_args()
try:
    model = load_network(args.loadfile)                    # ✅ disponible
    positions, labels = read_chess_file(args.chessfile)    # ⏳ issue #2
    X = encode_positions(positions)                        # ⏳ issue #2
    if args.mode == "predict":
        for label in model.predict(X):                     # ✅ disponible
            print(label)
    else:
        y = encode_labels(labels, model.classes)           # ⏳ issue #2
        if model.seed is not None:
            np.random.seed(model.seed)                     # reproductibilité
        model.fit(X, y, **model.training_config)           # ✅ disponible
        save_network(model, args.savefile or args.loadfile)  # ✅ disponible, écriture atomique
except ValueError as exc:
    exit_error(str(exc))
```

Points d'attention pour #1 :
- en `--predict`, **rien d'autre** que les prédictions sur stdout (les logs de `fit(verbose=True)` vont déjà sur stderr) ;
- `--train` doit refuser (code 84) une ligne sans label ;
- les labels du fichier doivent être adaptés aux classes du réseau chargé (ex. `Checkmate White` → `Checkmate` pour un réseau intermédiaire, `Check` pour un réseau basique) ;
- protéger l'import de `mytorch` comme dans le générateur, pour sortir en 84 si NumPy manque.

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

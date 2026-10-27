# 07 — Les tests

## Lancer les tests

```bash
.venv/bin/pytest tests/ -v                          # tout
.venv/bin/pytest tests/test_layers.py -v            # un fichier
.venv/bin/pytest tests/ -k gradient -v              # les tests dont le nom contient "gradient"
.venv/bin/python3 -m unittest discover -s tests -v  # sans pytest
```

Les tests sont écrits avec `unittest` (bibliothèque standard) et se lancent aussi avec `pytest`. 115 tests au total, environ 4 secondes.

## Ce que couvre chaque fichier

| Fichier | Tests | Ce qui est vérifié |
| :--- | :--- | :--- |
| `test_cli.py` | 15 | Arguments de l'analyseur : `--help` (code 0), modes manquants / multiples, `--save` hors `--train`, fichiers absents / vides, option inconnue → code 84 |
| `test_initializers.py` | 10 | Forme, type `float32`, moyenne et variance attendues de chaque initialiseur ; alias ; erreurs ; classe `Layer` abstraite |
| `test_layers.py` | 18 | Gradient check de `Linear` et de chaque activation ; valeurs connues ; stabilité de Sigmoid / Softmax ; Dropout (taux, mise à l'échelle, masque réutilisé au backward, identité en inférence, MC Dropout, reproductibilité) |
| `test_losses.py` | 9 | Valeurs de référence (ex. loss = $\log 5$ pour 5 classes équiprobables), logits extrêmes, gradient check des deux losses |
| `test_optimizers.py` | 9 | Formules exactes d'un pas, convergence sur une fonction quadratique, effet du weight decay, premier pas d'Adam ≈ `lr`, configs invalides |
| `test_xor.py` | 9 | **Test d'acceptation de #4** : XOR résolu (100 %, loss < 0.02) avec Adam et SGD ; validation ; early stopping qui restaure les meilleurs poids ; reproductibilité ; régression MSE |
| `test_normalization.py` | 14 | Standardisation, features constantes, gradient check, `state_dict`, adaptation sur la partie entraînement uniquement, démonstration sur données mal échelonnées |
| `test_config.py` | 12 | Valeurs par défaut, ~30 configurations invalides, message d'erreur précis, graine, aller-retour `get_config` → `build_from_config`, configs fournies valides |
| `test_serialization.py` | 12 | Aller-retour save/load avec prédictions **identiques**, réseau vierge, écrasement, permissions, métadonnées, fichiers corrompus / tronqués / étrangers, pickle refusé |
| `test_generator.py` | 7 | Le générateur lancé **comme le vrai exécutable** : fichiers créés, graines, noms répétés, `--help`, codes 84 sans aucun fichier écrit |

`tests/gradcheck.py` n'est pas un fichier de tests mais un **outil** partagé.

## Le gradient check

C'est le test le plus important de la librairie : il prouve que chaque `backward()` calcule les bons gradients. Une erreur de signe ou de transposée dans un `backward()` ne fait pas planter le programme : le réseau apprend juste mal, et c'est très difficile à repérer autrement.

### Principe (`tests/gradcheck.py`)

1. On tire un gradient aléatoire $G$ et on définit une loss artificielle $L = \sum (\text{forward}(X) \odot G)$. Sa dérivée par rapport à la sortie vaut exactement $G$.
2. **Gradient analytique** : `backward(G)` donne $\frac{\partial L}{\partial X}$ ; `parameters()` donne les gradients des poids.
3. **Gradient numérique** : pour chaque nombre $x$ (de l'entrée ou des poids), on calcule
   $\frac{L(x + \varepsilon) - L(x - \varepsilon)}{2\varepsilon}$ avec $\varepsilon = 10^{-6}$.
4. On compare avec l'**erreur relative** $\frac{\lVert a - n \rVert}{\lVert a \rVert + \lVert n \rVert}$, qui doit être $< 10^{-5}$.

| Fonction | Usage |
| :--- | :--- |
| `check_layer_gradients(layer, X)` | Renvoie la liste des erreurs relatives : entrée, puis chaque paramètre |
| `check_loss_gradient(loss, y_pred, y_true)` | Erreur relative du `backward()` d'une loss |
| `numerical_gradient(f, x)` | Différences finies centrées de `f()` par rapport à `x` |
| `relative_error(a, n)` | Erreur relative, robuste quand les deux gradients sont ~0 |

**Deux précautions** :
- Les calculs se font en **`float64`** : la précision du `float32` (~7 chiffres) est insuffisante pour mesurer une erreur de $10^{-5}$ avec $\varepsilon = 10^{-6}$. Les tests convertissent donc poids et entrées en `float64`.
- Pour ReLU, les entrées sont éloignées de 0 (|x| ≥ 0.1) : en 0, la dérivée n'est pas définie et la différence finie pourrait « enjamber » le coude.

### Ajouter une nouvelle couche

```python
from tests.gradcheck import check_layer_gradients

def test_ma_couche_gradient_check(self):
    layer = MaCouche(...)                              # paramètres convertis en float64 si besoin
    X = np.random.default_rng(0).standard_normal((4, 6))
    for error in check_layer_gradients(layer, X):
        self.assertLess(error, 1e-5)
```

## À savoir sur certains tests

- **XOR et graine** : le test utilise une graine fixe, il est donc déterministe. Sur 50 graines au hasard, environ 88 % des entraînements réussissent : avec seulement 8 neurones cachés, certaines initialisations produisent des ReLU « mortes ». C'est normal pour un réseau aussi petit, pas un bug.
- **Test du générateur** : il recopie `mytorch/`, `analyzer/` et `scripts/generator.py` dans un dossier temporaire, comme le fait `make`, puis lance le script avec `subprocess`. On teste ainsi le vrai comportement : imports, codes de sortie, fichiers écrits dans le dossier courant.
- **Early stopping** : le test donne au modèle une validation aux labels **inversés**. Plus le réseau apprend, plus la validation empire : l'arrêt doit se déclencher, et les poids restaurés doivent redonner exactement la meilleure `val_loss` enregistrée.
- **Pickle refusé** : un test glisse un objet Python dans un `.nn` et vérifie que le chargement est refusé, ce qui prouve qu'un fichier piégé ne peut pas exécuter de code.

## Conventions pour écrire un test

- Un fichier `tests/test_<module>.py`, une classe `unittest.TestCase` par composant.
- En tête de fichier, le même bloc `sys.path.insert(...)` que les autres tests, pour pouvoir lancer le fichier seul.
- Fixer la graine (`np.random.seed(...)` dans `setUp`) pour tout ce qui est aléatoire.
- Fichiers temporaires : `tempfile.TemporaryDirectory()`, nettoyé dans `tearDown`.
- Pour les cas d'erreur, `self.subTest(nom)` permet de tester beaucoup de cas dans une seule méthode tout en sachant lequel échoue.

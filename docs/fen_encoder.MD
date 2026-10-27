# Encodage FEN → entrée du réseau

Ce document explique comment [`analyzer/fen_encoder.py`](../analyzer/fen_encoder.py) transforme une position d'échecs (notation FEN) en vecteur de nombres exploitable par le réseau, avec des exemples détaillés case par case. La référence de l'API se trouve dans [ARCHITECTURE.md](../ARCHITECTURE.md).

---

## 1. Place dans le pipeline

L'encodeur fait le lien entre le fichier texte et le réseau :

```text
chessboards.txt ──read_chess_file──▶ fens + labels  (listes de str)
                                       │        │
                                  encode_fens  encode_labels
                                       │        │
                                       ▼        ▼
                                X (N, 773)   Y (N, nb_classes)
                                       │        │
                    --train :  model.fit(X, Y)  → le réseau ajuste ses poids
                    --predict: model.predict(X) → ["Check White", "Nothing", …]
```

Côté analyseur, l'utilisation ressemble à ceci :

```python
fens, labels = read_chess_file(args.chessfile, require_label=(args.mode == "train"))
X = encode_fens(fens)                         # matrice (N, 773) de 0 et de 1
if args.mode == "train":
    Y = encode_labels(labels, model.classes)  # matrice one-hot (N, nb_classes)
    model.fit(X, Y, epochs=..., batch_size=...)
else:
    for prediction in model.predict(X):       # une ligne par position, dans l'ordre
        print(prediction)
```

Chaque **ligne** de `X` est une position, chaque **colonne** est une des 773 questions oui/non décrites ci-dessous.

---

## 2. Structure du vecteur (773 valeurs)

| Indices | Contenu |
| :--- | :--- |
| `0 – 767` | 12 plans one-hot de 64 cases : « y a-t-il telle pièce sur telle case ? » |
| `768` | Trait : `1.0` si les blancs jouent, `0.0` sinon |
| `769 – 772` | Droits de roque `K`, `Q`, `k`, `q` |

### Numérotation des cases

Les cases sont numérotées dans l'ordre de lecture du FEN : de la rangée 8 vers la rangée 1, de gauche à droite.

```text
        a   b   c   d   e   f   g   h
   8 |  0   1   2   3   4   5   6   7
   7 |  8   9  10  11  12  13  14  15
   6 | 16  17  18  19  20  21  22  23
   5 | 24  25  26  27  28  29  30  31
   4 | 32  33  34  35  36  37  38  39
   3 | 40  41  42  43  44  45  46  47
   2 | 48  49  50  51  52  53  54  55
   1 | 56  57  58  59  60  61  62  63
```

### Plans de pièces

Les pièces suivent l'ordre de `PIECES = "PNBRQKpnbrqk"` (majuscules = blancs, minuscules = noirs) :

| Pièce | P | N | B | R | Q | K | p | n | b | r | q | k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n° de plan | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| indices | 0–63 | 64–127 | 128–191 | 192–255 | 256–319 | 320–383 | 384–447 | 448–511 | 512–575 | 576–639 | 640–703 | 704–767 |

D'où la formule : **index = n° de plan × 64 + n° de case**.

---

## 3. Exemple 1 : deux rois seuls

5ᵉ position de l'énoncé :

```text
8/8/8/8/8/8/8/k1K5 w - - 0 1
```

```text
        a   b   c   d   e   f   g   h
   8 |  .   .   .   .   .   .   .   .
  …     (rangées 7 à 2 vides)
   1 |  k   .   K   .   .   .   .   .
```

**Lecture du plateau, caractère par caractère** (`square` = compteur de case, démarre à 0) :

| Caractère | Ce que fait le code | `square` après |
|---|---|---|
| `8` (×7, avec les `/`) | 7 rangées vides : saute 8 cases à chaque fois | 56 |
| `k` | roi noir, plan 11, case 56 (a1) → **index 11×64+56 = 760** mis à 1 | 57 |
| `1` | 1 case vide (b1), on saute | 58 |
| `K` | roi blanc, plan 5, case 58 (c1) → **index 5×64+58 = 378** mis à 1 | 59 |
| `5` | 5 cases vides (d1 à h1) | 64 |

**Champs suivants :**
- `w` → les blancs jouent → **index 768 = 1**
- `-` → aucun roque → indices 769 à 772 restent à 0

**Résultat :** un vecteur de 773 valeurs avec **seulement trois 1**, aux indices **378, 760 et 768**. Tout le reste vaut 0.

---

## 4. Exemple 2 : échec à la tour

```text
4k3/8/8/8/8/8/8/4R1K1 b - - 0 1
```

```text
        a   b   c   d   e   f   g   h
   8 |  .   .   .   .   k   .   .   .
  …     (rangées 7 à 2 vides)
   1 |  .   .   .   .   R   .   K   .
```

La tour blanche en e1 attaque le roi noir en e8 par la colonne e, et ce sont les noirs qui jouent : le label est `Check White` (la couleur est celle du camp qui donne l'échec).

| Pièce | Case | Calcul | Index |
|---|---|---|---|
| `k` roi noir | e8 = 4 | 11 × 64 + 4 | **708** |
| `R` tour blanche | e1 = 60 | 3 × 64 + 60 | **252** |
| `K` roi blanc | g1 = 62 | 5 × 64 + 62 | **382** |
| trait `b` | — | index 768 | reste **0** |

**Résultat :** les 1 sont aux indices **252, 382 et 708**, tout le reste vaut 0.

Les deux exemples se vérifient directement :

```bash
python3 -c "
from analyzer.fen_encoder import encode_fen
import numpy as np
for fen in ['8/8/8/8/8/8/8/k1K5 w - - 0 1', '4k3/8/8/8/8/8/8/4R1K1 b - - 0 1']:
    print(fen, np.nonzero(encode_fen(fen))[0].tolist())
"
# 8/8/8/8/8/8/8/k1K5 w - - 0 1 [378, 760, 768]
# 4k3/8/8/8/8/8/8/4R1K1 b - - 0 1 [252, 382, 708]
```

---

## 5. Ce que le réseau en fait

Pour l'exemple 2, un neurone de la première couche `Linear` calcule :

```text
w₂₅₂·1 + w₃₈₂·1 + w₇₀₈·1 + (tous les autres poids × 0) + b
= w[tour blanche en e1] + w[roi blanc en g1] + w[roi noir en e8] + b
```

Les 0 annulent tous les autres poids : **seules les pièces présentes comptent**. Pendant l'entraînement, sur des milliers d'exemples, un neurone peut apprendre que « `w[tour blanche en e1]` et `w[roi noir en e8]` élevés ensemble » signifie souvent *Check*. Les couches suivantes apprennent les exceptions, par exemple une pièce posée entre les deux sur la colonne e qui bloque l'échec.

C'est pour cela qu'on utilise un one-hot par couple (pièce, case) plutôt qu'un code numérique par case : chaque couple a son propre poids, sans ordre artificiel entre les pièces (une dame ne « vaut » pas 5 pions pour le réseau).

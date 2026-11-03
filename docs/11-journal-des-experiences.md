# 11 — Journal des expériences

Ce journal garde la trace de **chaque entraînement significatif** : ce qui a été essayé, les résultats, ce qu'on en a appris et ce que ça a décidé pour la suite. Il sert à montrer la **progression** du projet en soutenance et à justifier les choix finaux ([10 — Hyperparamètres](10-hyperparametres.md)).

Chaque expérience doit être **reproductible** : le sujet impose de conserver les scripts d'entraînement. Chaque entrée donne donc le script qui la relance à l'identique (même données, même graine, mêmes résultats).

## Vue d'ensemble

| # | Date | Réseau | Données d'entraînement | Accuracy (test) | Ce qu'on a appris |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [01](#exp-01--premier-entraînement-du-réseau-complet) | 03/11/2026 | `full_network.conf` (5 classes) | 9 000 positions | **62,7 %** | Le pipeline fonctionne ; fort surapprentissage ; confusion échec / mat du même camp |
| [02](#exp-02--réseaux-pré-entraînés-sur-tout-le-dataset) | 03/11/2026 | basic / medium / full | 578 415 positions (tout le dataset) | **88,9 % / 85,5 % / 85,1 %** | Plus de données → le surapprentissage disparaît ; couleurs jamais confondues ; échec / mat reste l'erreur principale ; le basic est trop petit ; mats d'ouverture presque absents du dataset |

**Repères** pour lire une accuracy : au hasard, 50 % (basic), 33 % (medium), 20 % (full). Toujours répondre `Nothing` donne 33,3 % sur le test de l'exp 01 et 41,5 % sur celui de l'exp 02.

## Outils

| Outil | Rôle |
| :--- | :--- |
| `scripts/experiments/expNN_*.sh` | Relance une expérience de bout en bout ; sorties dans `experiments/output/expNN/` (ignoré par git) |
| `scripts/train_networks.py` | Entraîne les réseaux pré-entraînés `my_torch_network_*.nn` sur le dataset (exp 02) |
| `scripts/evaluate.py NETWORK PREDICTIONS CHESSFILE` | Compare les prédictions aux labels attendus : accuracy, précision / rappel / F1 par classe, matrice de confusion. Les labels sont adaptés aux classes du réseau, comme en `--train`. |

```bash
source .venv/bin/activate                      # NumPy doit être disponible
bash scripts/experiments/exp01_first_full_network.sh
```

---

## Exp 01 — Premier entraînement du réseau complet

**Date** : 03/11/2026 · **Script** : [`scripts/experiments/exp01_first_full_network.sh`](../scripts/experiments/exp01_first_full_network.sh) · **Durée** : ~6,5 s au total

### Objectif

Premier essai réel après l'arrivée du pipeline `my_torch_analyzer` (#1) et du dataset : vérifier que toute la chaîne **générateur → `--train` → `--predict`** fonctionne sur de vraies positions, et obtenir un **premier point de référence** (baseline) pour mesurer les progrès suivants.

### Protocole

| Élément | Valeur |
| :--- | :--- |
| Réseau | `configs/full_network.conf`, sans modification : 773 → 512 → 256 → 5, ReLU, dropout 0.3, Adam `lr=0.001`, `weight_decay=1e-4`, graine 42 — **528 901 paramètres** |
| Entraînement | Les 3 000 premières lignes de `dataset/{check,checkmate,nothing}/many_pieces.txt` → **9 000 positions** (environ 1,2 % du dataset) |
| Validation (early stopping) | 15 % de ces 9 000 positions, tirées par `fit()` (`val_split` du `.conf`), patience 7 |
| Test | Les lignes 3 001 à 4 000 des mêmes fichiers → **3 000 positions jamais vues** pendant l'entraînement |
| Commandes | `my_torch_generator full_network.conf 1`, puis `my_torch_analyzer --train --save trained.nn …`, puis `--predict`, puis `scripts/evaluate.py` |

Répartition du jeu de test :

| Nothing | Check White | Check Black | Checkmate White | Checkmate Black |
| :--- | :--- | :--- | :--- | :--- |
| 1 000 | 566 | 434 | 610 | 390 |

### Résultats

**Accuracy sur le test : 62,7 %** (1 880 / 3 000). Résumé de l'analyseur :

```
Trained on 9000 positions (11 epochs) - best epoch 4: val_loss 0.8956, val_acc 0.6163
```

**Courbe d'apprentissage** (log de `--train`, sur stderr) :

| Époque | `train_loss` | `train_acc` | `val_loss` | `val_acc` |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 1.540 | 67,4 % | 1.087 | 54,4 % |
| 2 | 0.906 | 82,2 % | 0.943 | 58,9 % |
| 3 | 0.690 | 87,3 % | 0.898 | 61,0 % |
| **4** | **0.553** | **92,2 %** | **0.896** ← minimum | **61,6 %** |
| 5 | 0.466 | 95,0 % | 0.921 | 61,2 % |
| 7 | 0.312 | 98,0 % | 1.043 | 62,7 % |
| 9 | 0.211 | 99,4 % | 1.125 | 62,9 % |
| 11 | 0.145 | 99,8 % | 1.265 | 62,7 % |

Arrêt après l'époque 11 (7 époques sans amélioration de `val_loss`), poids de l'**époque 4** restaurés.

**Métriques par classe** (`scripts/evaluate.py`) :

| Classe | Précision | Rappel | F1 | Effectif |
| :--- | :--- | :--- | :--- | :--- |
| Nothing | 0.702 | 0.747 | 0.724 | 1 000 |
| Check White | 0.480 | 0.486 | 0.483 | 566 |
| Check Black | 0.573 | 0.454 | 0.506 | 434 |
| Checkmate White | 0.654 | 0.644 | 0.649 | 610 |
| Checkmate Black | 0.641 | 0.687 | 0.663 | 390 |

**Matrice de confusion** (lignes = attendu, colonnes = prédit) :

| attendu ↓ / prédit → | Nothing | Check White | Check Black | Checkmate White | Checkmate Black |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Nothing** | **747** | 109 | 59 | 48 | 37 |
| **Check White** | 131 | **275** | 0 | 155 | 5 |
| **Check Black** | 122 | 6 | **197** | 3 | 106 |
| **Checkmate White** | 33 | 180 | 2 | **393** | 2 |
| **Checkmate Black** | 31 | 3 | 86 | 2 | **268** |

### Analyse

1. **La chaîne complète fonctionne** sur de vraies positions : génération, entraînement, sauvegarde, prédiction d'une ligne par position, évaluation. Le résultat est trois fois meilleur que le hasard (20 %) et presque deux fois meilleur que « toujours `Nothing` » (33 %) : le réseau a réellement appris quelque chose.

2. **Fort surapprentissage** : 99,8 % sur l'entraînement contre ~62 % en validation. La `val_loss` atteint son minimum dès l'**époque 4** puis remonte, alors que la `train_loss` continue de baisser : le réseau apprend les 9 000 positions par cœur au lieu de règles générales. C'est attendu : **528 901 paramètres pour 9 000 exemples**, soit environ 59 paramètres par exemple. L'**early stopping a joué son rôle** [OPT1] : il a gardé les poids de l'époque 4 au lieu de ceux de l'époque 11.

3. **La couleur est presque toujours juste** : `Check White` n'est jamais prédit `Check Black` (0) et inversement (6). Le réseau a bien appris **quel camp attaque**.

4. **La principale erreur est « échec ou mat ? » pour un même camp** : 155 `Check White` prédits `Checkmate White`, 180 dans l'autre sens ; 106 et 86 pour les noirs. Distinguer un échec d'un mat demande de savoir si le roi a une case de fuite ou si une pièce peut s'interposer : une règle plus fine que « une pièce attaque le roi », que 9 000 exemples ne suffisent pas à apprendre.

5. **Deuxième erreur : `Nothing` contre `Check`** (131 et 122 échecs prédits `Nothing`) : certaines attaques ne sont pas repérées. Hypothèse à vérifier : les attaques de pièces lointaines (fous, tours, dames) sont plus difficiles à détecter à partir d'un encodage case par case que celles des pièces proches.

### Limites de cette expérience

- Une seule source de positions (`many_pieces`) : les positions à 10 et 20 pièces ne sont pas testées.
- Un seul essai, une seule graine : l'écart de ±1 à 2 % n'est pas significatif (il faudrait une validation croisée).
- La validation interne de `fit()` n'est pas stratifiée.

### Décisions pour la suite

| Piste | Pourquoi | Outil |
| :--- | :--- | :--- |
| **Entraîner sur beaucoup plus de données** (des dizaines puis centaines de milliers de positions, les trois tailles de positions) | Cause principale du surapprentissage | Script d'entraînement utilisant la librairie (en `--train`, tout le fichier est encodé en mémoire : ~3 Ko par position) |
| **Régler la régularisation et le learning rate** (`layers.*.rate`, `optimizer.weight_decay`, `optimizer.lr`) | Le minimum de `val_loss` arrive très tôt | `grid_search` |
| **Comparer avec un réseau plus petit** (`medium` en largeur, 256 → 128) | Moins de paramètres → moins de mémorisation | `grid_search` avec la clé `layers` |
| **Mesurer par validation croisée** | Rendre les comparaisons fiables | `cross_validate`, ou `grid_search(k=5)` |
| Suivre le **rappel de `Check` et `Checkmate`** en plus de l'accuracy | Ce sont les erreurs principales | `scripts/evaluate.py` |

---

## Exp 02 — Réseaux pré-entraînés sur tout le dataset

**Date** : 03/11/2026 · **Script** : [`scripts/train_networks.py`](../scripts/train_networks.py) · **Durée** : 3 h 32 au total (basic 18 min, medium 70 min, full 2 h 04), 2 Go de RAM au maximum

### Objectif

Produire les réseaux **pré-entraînés exigés par le sujet** (`my_torch_network_basic.nn`, `_medium.nn`, `_full.nn`), évalués par l'autograder, et vérifier si l'exp 01 avait raison : le surapprentissage venait-il du **manque de données** ?

### Protocole

| Élément | Valeur |
| :--- | :--- |
| Réseaux | Les trois `.conf` fournis, **sans modification** (mêmes hyperparamètres qu'à l'exp 01 pour le full) |
| Données | **Tout le dataset** : 723 019 positions, 9 fichiers (`check`, `checkmate`, `nothing` × 10 pièces, 20 pièces, beaucoup de pièces) |
| Découpage | Stratifié sur les 5 labels, graine 42, **identique pour les trois réseaux** : 578 415 entraînement (80 %), 72 302 validation (10 %, early stopping), 72 302 **test** (10 %, jamais vu) |
| Labels | Adaptés à chaque réseau (`encode_labels`) : `Checkmate White` → `Checkmate` (medium) → `Check` (basic) |
| Commande | `python3 scripts/train_networks.py` (tout le dataset, les trois niveaux) |

Le script stocke les positions en `uint8` (560 Mo au lieu de 2,2 Go en `float32`). Pour tenir en mémoire, `Normalization.adapt` calcule maintenant ses statistiques par paquets ([03](03-couches.md#normalization-normalizationpy)).

### Résultats

| Réseau | Classes | Paramètres | Accuracy **test** | Meilleure époque | Arrêt |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `my_torch_network_basic.nn` | 2 | 99 330 | **88,9 %** | 27 | limite de 30 époques |
| `my_torch_network_medium.nn` | 3 | 231 427 | **85,5 %** | 38 | limite de 40 époques |
| `my_torch_network_full.nn` | 5 | 528 901 | **85,1 %** | 38 | early stopping (époque 45) |

Ces scores ont été **recalculés en passant par l'exécutable** (`my_torch_analyzer --predict` sur les 72 302 positions de test, puis `scripts/evaluate.py`) : mêmes chiffres. La prédiction des 72 302 positions prend 2,5 s (basic) à 4,6 s (full).

**Progression du full par rapport à l'exp 01** (même réseau, même config) : **62,7 % → 85,1 %**.

**Courbes d'apprentissage** (extraits de `experiments/output/train_networks/<niveau>_history.json`) :

| Époque | basic `val_acc` | medium `val_acc` | full `val_acc` |
| :--- | :--- | :--- | :--- |
| 1 | 78,8 % | 69,0 % | 70,4 % |
| 5 | 85,9 % | 80,1 % | 80,4 % |
| 10 | 87,3 % | 83,2 % | 82,9 % |
| 20 | 88,6 % | 85,0 % | 84,3 % |
| 30 | 88,9 % | 85,6 % | 84,7 % |
| 38 | — | **86,0 %** | **85,3 %** |

Écart entraînement / validation à la meilleure époque : basic 91,4 % / 89,1 %, medium 89,2 % / 86,0 %, full 88,4 % / 85,3 %.

**Métriques par classe sur le test** :

| Réseau | Classe | Précision | Rappel | F1 |
| :--- | :--- | :--- | :--- | :--- |
| basic | Nothing | 0.887 | 0.838 | 0.862 |
| basic | Check | 0.889 | 0.925 | 0.907 |
| medium | Nothing | 0.928 | 0.891 | 0.909 |
| medium | Check | 0.794 | 0.817 | 0.805 |
| medium | Checkmate | 0.825 | 0.849 | 0.837 |
| full | Nothing | 0.918 | 0.898 | 0.908 |
| full | Check White | 0.809 | 0.792 | 0.800 |
| full | Check Black | 0.785 | 0.797 | 0.791 |
| full | Checkmate White | 0.807 | 0.863 | 0.834 |
| full | Checkmate Black | 0.824 | 0.834 | 0.829 |

**Matrice de confusion du full** (test, lignes = attendu, colonnes = prédit) :

| attendu ↓ / prédit → | Nothing | Check White | Check Black | Checkmate White | Checkmate Black |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Nothing** | **26 931** | 1 350 | 1 274 | 254 | 191 |
| **Check White** | 1 127 | **10 738** | 0 | 1 694 | 0 |
| **Check Black** | 1 076 | 0 | **9 068** | 0 | 1 227 |
| **Checkmate White** | 103 | 1 186 | 0 | **8 141** | 0 |
| **Checkmate Black** | 106 | 0 | 1 211 | 0 | **6 625** |

**Positions d'exemple du sujet** (`my_torch_analyzer --predict` sur les 5 positions du PDF) :

| Position | Attendu (full) | basic | medium | full |
| :--- | :--- | :--- | :--- | :--- |
| Mat du berger (`rnb1kbnr/…/RNBQKBNR w KQkq - 1 3`) | Checkmate Black | ✅ Check | ❌ Nothing | ❌ Check Black |
| Position de départ | Nothing | ✅ | ✅ | ✅ |
| Après 1. d4 | Nothing | ✅ | ✅ | ✅ |
| Dame en h5, roi noir en échec | Check White | ✅ Check | ✅ Check | ✅ |
| Rois seuls (`8/8/8/8/8/8/8/k1K5`) | Nothing | ✅ | ✅ | ✅ |

**Le full sait-il répondre aux niveaux plus simples ?** En ramenant ses réponses aux classes moins détaillées (`Checkmate White` → `Checkmate` → `Check`) :

| Tâche | Réseau dédié | Full ramené à cette tâche |
| :--- | :--- | :--- |
| basic (2 classes) | 88,9 % | **92,4 %** |
| medium (3 classes) | **85,5 %** | 85,1 % |

### Analyse

1. **Le manque de données était bien la cause principale.** Avec 64 fois plus de positions d'entraînement qu'à l'exp 01, le full passe de 62,7 % à 85,1 %, et l'écart entraînement / validation tombe d'environ 37 points à environ 3 points : **le surapprentissage a presque disparu**.

2. **Les couleurs sont parfaitement séparées** : le full ne confond **jamais** un échec ou un mat blanc avec un noir (tous les « 0 » de la matrice). Toutes ses erreurs sont de deux types :
   - **échec ↔ mat du même camp** (1 694 + 1 186 côté blancs, 1 227 + 1 211 côté noirs) : c'est toujours l'erreur principale, comme à l'exp 01 ;
   - **rien ↔ échec** : environ 2 600 `Nothing` pris pour des échecs, environ 2 200 échecs manqués.

3. **Le basic et le medium n'avaient pas fini d'apprendre** : ils se sont arrêtés à leur limite d'époques (30 et 40) alors que leur meilleure époque était la 27 et la 38, avec une `val_loss` qui baissait encore très lentement. Le full, lui, s'est arrêté par early stopping : il a atteint son plateau.

4. **Le full ramené au niveau basic (92,4 %) bat le basic dédié (88,9 %).** Le full a plus de capacité (512 → 256 neurones contre 128) et apprend une tâche plus riche, ce qui l'aide aussi à repérer les échecs. Le basic est donc probablement **trop petit et pas assez entraîné**. Au niveau medium, les deux se valent (85,5 % contre 85,1 %).

5. **Le mat du berger, premier exemple du sujet, est raté par le medium et le full.** C'est un mat **d'ouverture**, avec presque toutes les pièces encore sur l'échiquier, en 2 coups. Le full a bien vu l'échec et la bonne couleur (`Check Black`), mais pas le mat. Hypothèse à vérifier : ce type de mat très précoce est rare dans le dataset, dont les mats viennent surtout de milieux et de fins de partie.

### Analyses complémentaires

Mesurées après coup sur le même jeu de test (72 302 positions), avec les réseaux poussés.

**Précision selon la source des positions** :

| Fichiers | Positions de test | basic | medium | full |
| :--- | :--- | :--- | :--- | :--- |
| `10_pieces` | 21 996 | 86,6 % | **87,0 %** | **87,2 %** |
| `20_pieces` | 30 107 | 89,4 % | 84,0 % | 83,2 % |
| `many_pieces` | 20 199 | **90,6 %** | 86,3 % | 85,6 % |

**Full, selon le nombre de pièces sur l'échiquier** :

| Pièces | Positions | Accuracy |
| :--- | :--- | :--- |
| 2 – 8 | 17 694 | 87,4 % |
| 9 – 12 | 12 131 | 85,2 % |
| 13 – 20 | 26 926 | **82,6 %** |
| 21 – 26 | 13 096 | 85,1 % |
| 27 – 32 | 2 455 | 94,1 % (75 % de `Nothing` dans ce groupe : chiffre flatteur) |

**Mats selon le moment de la partie** (full, mats du test uniquement) :

| Numéro de coup | Mats | Accuracy |
| :--- | :--- | :--- |
| 1 – 20 | 637 | 80,1 % |
| 21 – 40 | 7 528 | 81,0 % |
| 41 – 60 | 5 886 | 86,8 % |
| 61 et plus | 3 321 | **91,8 %** |

**Les mats d'ouverture sont presque absents du dataset** : sur 173 719 mats, seuls **396 (0,23 %)** ont lieu avant le 10ᵉ coup, et seuls **1 087** ont 28 pièces ou plus sur l'échiquier.

**Ce que ça montre** :
- **L'hypothèse sur le mat du berger est confirmée.** Les mats précoces, avec beaucoup de pièces, sont très rares dans les données. Le réseau en a très peu vu, et il reconnaît d'autant moins bien un mat qu'il arrive tôt dans la partie (80 % avant le coup 20, contre 92 % après le coup 60).
- **Les positions de milieu de partie (13 à 20 pièces) sont les plus difficiles** pour le full : il y a assez de pièces pour bloquer ou défendre, ce qui rend la frontière échec / mat subtile.
- **Pour distinguer échec et mat** (medium, full), les fins de partie à 10 pièces sont les plus faciles. Pour **repérer un échec** (basic), ce sont au contraire les positions chargées.

### Décisions pour la suite

| Piste | Pourquoi |
| :--- | :--- |
| **Pousser ces trois réseaux** dans le dépôt | Le sujet l'exige ; ce sont nos meilleurs réseaux à ce jour |
| **Agrandir le basic** (par ex. 512 → 256, comme le full) et **augmenter `epochs`** pour basic et medium | Le full ramené au basic fait 3,5 points de mieux ; basic et medium se sont arrêtés à la limite d'époques |
| **Travailler la différence échec / mat** | Première source d'erreurs à tous les niveaux. Pistes : plus de capacité, plus d'époques, réglage du learning rate (planning), vérifier la répartition des mats dans le dataset |
| **Ajouter des mats d'ouverture au dataset** (mat du berger, mat du lion…) | Vérifié : seuls 0,23 % des mats ont lieu avant le 10ᵉ coup, et le premier exemple du sujet est raté |
| Grid search sur `lr`, dropout, `weight_decay` | Les configs n'ont pas encore été réglées ; chaque essai complet coûte 20 min à 2 h, donc faire la recherche sur une partie du dataset |

---

## Modèle pour une nouvelle entrée

Copier ce bloc, numéroter l'expérience, et ajouter une ligne au tableau de la vue d'ensemble.

```markdown
## Exp NN — Titre court

**Date** : JJ/MM/AAAA · **Script** : `scripts/experiments/expNN_….sh` · **Durée** : …

### Objectif
Ce qu'on cherche à vérifier ou à améliorer, par rapport à quelle expérience.

### Protocole
Réseau (config et modifications), données d'entraînement / validation / test, graine, commandes.

### Résultats
Accuracy test, courbe d'apprentissage, métriques par classe, matrice de confusion.

### Analyse
Ce que les chiffres montrent, comparé aux expériences précédentes.

### Décisions pour la suite
Ce qu'on garde, ce qu'on change, la prochaine expérience.
```

# 11 — Journal des expériences

Ce journal garde la trace de **chaque entraînement significatif** : ce qui a été essayé, les résultats, ce qu'on en a appris et ce que ça a décidé pour la suite. Il sert à montrer la **progression** du projet en soutenance et à justifier les choix finaux ([10 — Hyperparamètres](10-hyperparametres.md)).

Chaque expérience doit être **reproductible** : le sujet impose de conserver les scripts d'entraînement. Chaque entrée donne donc le script qui la relance à l'identique (même données, même graine, mêmes résultats).

## Vue d'ensemble

| # | Date | Réseau | Données d'entraînement | Accuracy (test) | Ce qu'on a appris |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [01](#exp-01--premier-entraînement-du-réseau-complet) | 03/11/2026 | `full_network.conf` (5 classes) | 9 000 positions | **62,7 %** | Le pipeline fonctionne ; fort surapprentissage ; confusion échec / mat du même camp |

**Repères** pour lire une accuracy sur 5 classes : répondre au hasard donne 20 % ; répondre toujours `Nothing` (la classe la plus fréquente du test) donne 33,3 %.

## Outils

| Outil | Rôle |
| :--- | :--- |
| `scripts/experiments/expNN_*.sh` | Relance une expérience de bout en bout ; sorties dans `experiments/output/expNN/` (ignoré par git) |
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

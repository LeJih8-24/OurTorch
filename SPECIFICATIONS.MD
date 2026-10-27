
# Projet MY_TORCH — Analyseur d'échiquier par réseau de neurones

> *A Clash of Kings* — Spécifications v2.6.0

## Directives pour l'Assistant IA

Ce document constitue le cahier des charges de référence. Lors de toute intervention sur ce dépôt (lecture de code, génération de correctifs, architecture), vous devez impérativement respecter les contraintes techniques ci-dessous. Toute nouvelle implémentation doit correspondre aux exigences de ce document.

Les mots **DOIT** (MUST), **DEVRAIT** (SHOULD) et **PEUT** (MAY) ont le sens habituel : obligatoire, fortement recommandé, optionnel.

## Contexte

Le but est de développer une « arme intelligente » capable de prédire l'issue d'une bataille. Elle est testée à petite échelle sur un problème dont la vérité terrain est facile à calculer : déterminer l'état d'une partie d'échecs à partir d'une position.

Le projet se compose de deux parties :

1. **Un module de réseau de neurones** (`NN module`) : bibliothèque générique pour créer, entraîner, sauvegarder, charger et analyser un réseau de neurones.
2. **Un exécutable d'analyse** (`my_torch_analyzer`) : utilise le module pour entraîner un réseau ou prédire l'état de positions d'échecs.

## Contraintes Techniques Strictes

* **Nom du binaire :** `my_torch_analyzer`.
* **Langage :** libre, du moment qu'il fonctionne sur « the dump » (l'environnement de correction).
* **Compilation :** si nécessaire, via un `Makefile` contenant les règles `re`, `clean` et `fclean`.
* **Interdiction des bibliothèques de réseaux de neurones :** PyTorch, TensorFlow, Keras, JAX, scikit-learn (partie réseaux) ou toute bibliothèque équivalente sont **strictement interdits**. Le réseau, la rétropropagation et les optimiseurs doivent être implémentés à la main. Les bibliothèques de calcul numérique de base (ex. NumPy) et de visualisation (ex. matplotlib) restent utilisables pour l'algèbre linéaire et les graphiques.
* **Apprentissage supervisé obligatoire :** la solution DOIT reposer sur du machine learning entraîné par apprentissage supervisé. Un algorithme d'échecs classique ne peut pas servir de prédicteur (il peut en revanche servir à **générer les labels** du jeu d'entraînement).
* **Gestion des erreurs :** tout message d'erreur DOIT être écrit sur la sortie d'erreur (`stderr`) et le programme DOIT alors quitter avec le code **84**. Code de retour **0** en l'absence d'erreur.
* **Livraison :**
  * Tous les fichiers sources DOIVENT être livrés, à l'exception des fichiers inutiles (binaires, fichiers temporaires, fichiers objets…).
  * Les fichiers bonus (y compris un éventuel `Makefile` spécifique) DOIVENT se trouver dans un répertoire `bonus/`.

## Livrables Obligatoires

| Livrable                         | Statut                | Détail                                                                                                                                                                     |
| -------------------------------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Module NN                        | DOIT                  | Création, sauvegarde/chargement, entraînement d'un réseau                                                                                                                |
| `my_torch_analyzer`            | DOIT                  | Modes `--train` et `--predict`                                                                                                                                          |
| Réseau pré-entraîné          | DOIT                  | Poussé dans le dépôt, chargeable par l'analyseur, nom commençant par `my_torch_network` (ex. `my_torch_network_full.nn`). Évalué automatiquement par l'autograder |
| Scripts d'entraînement          | DOIT                  | Tous les scripts utilisés pour générer et entraîner le réseau doivent être conservés, afin de pouvoir reproduire l'entraînement à l'identique                      |
| Jeux de données d'entraînement | DOIT être conservés | Pas d'obligation de les pousser (volumineux), mais ils DOIVENT pouvoir être présentés en soutenance                                                                      |
| `README`                       | DOIT                  | Documentation du projet                                                                                                                                                     |
| Benchmarks documentés           | DEVRAIT               | Avec preuves visuelles (graphiques). Un benchmark sans visuel ne vaut rien                                                                                                  |
| Justification des choix          | DEVRAIT               | Architecture, hyperparamètres, format des fichiers                                                                                                                         |

## Module de Réseau de Neurones

Le module DOIT pouvoir être chargé dynamiquement ou importé directement selon le langage choisi.

### Fonctionnalités minimales (obligatoires)

* **[NN1] Création** d'un nouveau réseau de neurones.
* **[NN2] Sauvegarde / chargement** d'un réseau depuis/vers un fichier.
* **[NN3] Entraînement** d'un réseau.

### Hyperparamètres

Plus le module expose de réglages, mieux c'est. Liste indicative :

* Topologie : nombre de couches, nombre de neurones par couche.
* Fonctions d'activation par couche (ReLU, sigmoïde, tanh, softmax en sortie…).
* Fonction de coût (entropie croisée, MSE…).
* Taux d'apprentissage et éventuel planning (decay, warmup…).
* Taille de batch, nombre d'époques.
* Optimiseur (SGD, momentum, Adam…).
* Régularisation (L2 / weight decay, dropout).
* Stratégie d'initialisation des poids.
* Graine aléatoire (reproductibilité).

### Génération de réseaux vierges

Il est conseillé de pouvoir générer facilement des réseaux non entraînés pour tester différentes configurations, par exemple :

* **[GEN1] Générateur depuis fichier de configuration** (exemple d'interface) :

  ```
  ./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]
  ```

  * `config_file_i` : fichier décrivant le réseau à générer.
  * `nb_i` : nombre de réseaux à générer à partir de cette configuration.
  * Exemple : `./my_torch_generator basic_network.conf 3` produit `basic_network_1.nn`, `basic_network_2.nn`, `basic_network_3.nn`.
* **[GEN2] Génération directement dans le code** via l'API du module.
* **[GEN3]** Toute autre approche pertinente.

### Formats de fichiers

Le format des fichiers de configuration et de sauvegarde des réseaux est **libre**. Il faut toutefois réfléchir aux informations réellement nécessaires (topologie, activations, poids, biais, normalisation des entrées, encodage des sorties, éventuellement état de l'optimiseur) et pouvoir justifier ce choix.

## L'Analyseur (`my_torch_analyzer`)

### Usage

```
./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE
```

| Argument            | Description                                                                                                                                                                                                                                   |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--train`         | Mode entraînement. Chaque ligne de `CHESSFILE` contient une position FEN suivie de la sortie attendue, séparées par un espace. Le réseau entraîné est sauvegardé dans `SAVEFILE` si `--save` est fourni, sinon dans `LOADFILE` |
| `--predict`       | Mode prédiction. Chaque ligne de `CHESSFILE` contient une position FEN, et éventuellement une sortie attendue                                                                                                                             |
| `--save SAVEFILE` | Sauvegarde le réseau dans `SAVEFILE`. Valide uniquement en mode entraînement                                                                                                                                                              |
| `LOADFILE`        | Fichier contenant un réseau de neurones                                                                                                                                                                                                      |
| `CHESSFILE`       | Fichier contenant les positions d'échecs                                                                                                                                                                                                     |

`--help` DOIT afficher l'usage. Toute erreur d'argument (mode manquant ou multiple, `--save` hors mode train, fichier introuvable ou illisible, FEN invalide, réseau corrompu…) DOIT produire un message sur `stderr` et un code de sortie 84.

### Entrée : notation FEN

Les positions sont décrites en [notation Forsyth–Edwards](https://fr.wikipedia.org/wiki/Notation_Forsyth-Edwards) (6 champs : placement des pièces, trait, roques, prise en passant, demi-coups, numéro de coup).

```
rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
```

En mode entraînement, le label suit la FEN sur la même ligne, ex. :

```
rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3 Checkmate White
```

### Sortie : état de la partie

Le réseau classe chaque position parmi :

* **`Checkmate`** : un joueur a gagné.
* **`Check`** : un roi est en échec.
* **`Nothing`** : rien de particulier.
* **`Stalemate`** *(bonus)* : pat — peu de configurations existent.

Niveaux de précision possibles (selon le réseau chargé) :

| Niveau         | Exemple de sortie                                                               |
| -------------- | ------------------------------------------------------------------------------- |
| Basique        | `Check` / `Nothing`                                                         |
| Intermédiaire | `Checkmate` / `Check` / `Nothing`                                         |
| Complet        | `Checkmate Black` / `Check White` / `Nothing` (couleur du camp concerné) |

### Règles du mode prédiction

* Le programme DOIT analyser **toutes** les positions du fichier.
* Les prédictions DOIVENT être affichées sur la sortie standard, **une par ligne, dans le même ordre** que les positions d'entrée.
* Aucune autre sortie ne doit polluer `stdout` (l'autograder compare ligne à ligne).

```
$> ./my_torch_analyzer --predict my_torch_network_full.nn chessboards.txt | head -n 5
Checkmate Black
Nothing
Nothing
Check White
Nothing
```

En mode entraînement, aucune contrainte sur l'affichage.

## Analyse et Optimisation

Les méthodes suivantes doivent pouvoir être démontrées en soutenance.

* **[OPT1] Lutte contre le surapprentissage (obligatoire) :** le processus d'entraînement DOIT chercher à éviter l'overfitting (séparation train/validation/test, early stopping, régularisation L2, dropout, augmentation de données…).
* **[OPT2] Optimisation des hyperparamètres :** éviter de rester bloqué dans un minimum local. Au-delà du réglage manuel du taux d'apprentissage et de l'aléatoire, s'appuyer sur des méthodes d'optimisation d'hyperparamètres et des métaheuristiques (grid/random search, algorithmes génétiques, recuit simulé…).
* **[OPT3] Efficacité de l'apprentissage :** liste non exhaustive de techniques :
  * MC dropout ;
  * descente de gradient stochastique / par mini-batch ;
  * stratégies d'initialisation des poids (LeCun, Xavier/Glorot, He…) ;
  * etc.

## Documentation

* Un **README** est obligatoire.
* Des **benchmarks** DEVRAIENT être fournis, **avec des preuves visuelles** (courbes d'apprentissage, matrices de confusion, comparaisons de configurations).
* Tout document permettant de justifier les choix de conception (architecture, hyperparamètres, encodage des entrées, format de fichier) est bienvenu. Plus la documentation est utile et pertinente, mieux c'est.

## Bonus (« Going further »)

* **[B1] Parallélisme :** accélérer l'apprentissage (multithreading, multicœur, GPGPU…).
* **[B2] Comparaison de modèles :** afficher plusieurs courbes d'apprentissage sur un même graphique.
* **[B3] Métriques multiples :** évaluer l'apprentissage selon plusieurs métriques (accuracy, précision, rappel, F1 par classe…).
* **[B4] Gestion du pat** (`Stalemate`).
* **[B5] IA d'échecs complète** capable de jouer.

Tous les fichiers bonus DOIVENT se trouver dans `bonus/`.

## Critères d'Acceptation

- [ ] `make`, `make re`, `make clean`, `make fclean` fonctionnent (si compilation nécessaire).
- [ ] Aucune bibliothèque de réseaux de neurones interdite n'est utilisée.
- [ ] Le module permet de créer, entraîner, sauvegarder et charger un réseau.
- [ ] `./my_torch_analyzer --help` affiche l'usage.
- [ ] `--predict` produit une ligne par position, dans l'ordre, sur `stdout`.
- [ ] `--train` sauvegarde dans `SAVEFILE` ou, à défaut, dans `LOADFILE`.
- [ ] Les erreurs sont écrites sur `stderr` avec le code de sortie 84.
- [ ] Un réseau pré-entraîné `my_torch_network*` est présent dans le dépôt et chargeable.
- [ ] Les scripts d'entraînement sont versionnés et permettent de reproduire le réseau.
- [ ] Les jeux de données sont conservés et présentables en soutenance.
- [ ] Une mesure anti-overfitting est en place et démontrable.
- [ ] Le README et les benchmarks illustrés sont fournis.

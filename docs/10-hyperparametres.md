# 10 — Les hyperparamètres

Ce chapitre regroupe **tous les réglages** de `mytorch` : leur rôle, leurs valeurs possibles, l'effet d'une valeur trop haute ou trop basse, et la valeur retenue dans nos configurations avec sa justification.

> ⚠️ **État des justifications** : les valeurs actuelles des `.conf` sont des **points de départ** reconnus (valeurs usuelles de la littérature et des frameworks), pas encore **mesurés** sur notre jeu de données. La justification définitive viendra des expériences de la section [7](#7-résultats-expérimentaux-à-compléter), à remplir avec la recherche d'hyperparamètres [OPT2] et les benchmarks (issue #6). En soutenance, il faudra pouvoir montrer ces mesures.

---

## 1. Paramètre ou hyperparamètre ?

| | Paramètres | Hyperparamètres |
| :--- | :--- | :--- |
| Exemples | Poids `W` et biais `b` des couches `Linear` | Learning rate, nombre de couches, taux de dropout, taille de batch… |
| Fixés par | **L'entraînement** (l'optimiseur les modifie à chaque pas) | **Nous**, avant l'entraînement |
| Où | Dans le `.nn` | Dans le `.conf` (puis recopiés dans le `.nn`) |
| Combien | ~100 000 à ~530 000 selon le réseau | Une vingtaine |

Les hyperparamètres décident **comment** le réseau apprend ; les paramètres sont **ce qu'il apprend**. On ne peut pas les régler par descente de gradient : il faut essayer plusieurs valeurs et comparer les résultats **sur la validation**.

---

## 2. Vue d'ensemble

Tous se règlent dans le `.conf` ([format complet](05-configuration-et-sauvegarde.md#1-les-fichiers-conf-configpy)). La colonne « Défaut » donne la valeur utilisée quand la clé est absente.

| Hyperparamètre | Clé dans le `.conf` | Valeurs possibles | Défaut | Rôle |
| :--- | :--- | :--- | :--- | :--- |
| Nombre de couches cachées | nombre de `linear` dans `layers` | ≥ 0 couche cachée | — | Profondeur, capacité |
| Neurones par couche | `layers[i].out` | entier > 0 | — | Largeur, capacité |
| Activation | `layers[i].type` | `relu`, `leakyrelu`, `sigmoid`, `tanh` | — | Non-linéarité |
| Pente de LeakyReLU | `layers[i].alpha` | ≥ 0 | `0.01` | Gradient pour $x < 0$ |
| Initialisation | `layers[i].init` | `he`, `xavier`, `xavier_uniform`, `lecun` | `he` | Échelle des poids de départ |
| Normalisation des entrées | couche `normalization` en premier | présente / absente | — | Met les features à la même échelle |
| Taux de dropout | `layers[i].rate` | $[0, 1)$ | `0.2` | Régularisation |
| MC Dropout | `layers[i].mc_dropout` | `true` / `false` | `false` | Dropout actif en prédiction |
| Fonction de coût | `loss` | `cross_entropy`, `mse` | `cross_entropy` | Mesure de l'erreur |
| Optimiseur | `optimizer.type` | `adam`, `sgd` | `adam` | Règle de mise à jour des poids |
| Learning rate | `optimizer.lr` | > 0 | Adam `0.001`, SGD `0.01` | Taille des pas |
| Momentum (SGD) | `optimizer.momentum` | $[0, 1)$ | `0.9` | Inertie des mises à jour |
| $\beta_1$, $\beta_2$, $\epsilon$ (Adam) | `optimizer.beta1`, `beta2`, `eps` | $[0, 1)$, $[0, 1)$, > 0 | `0.9`, `0.999`, `1e-8` | Mémoire des moyennes d'Adam |
| Régularisation L2 | `optimizer.weight_decay` | ≥ 0 | `0` | Pénalise les grands poids |
| Époques | `training.epochs` | entier > 0 | `30` | Durée maximale |
| Taille de batch | `training.batch_size` | entier > 0 | `128` | Exemples par pas |
| Part de validation | `training.val_split` | $[0, 1)$ | `0.15` | Données mises de côté pour mesurer |
| Patience | `training.patience` | entier > 0 ou `null` | `5` | Early stopping |
| Graine | `seed` | entier ≥ 0 ou `null` | `null` | Reproductibilité |

---

## 3. Valeurs retenues dans nos configurations

| | `basic_network.conf` | `medium_network.conf` | `full_network.conf` |
| :--- | :--- | :--- | :--- |
| Classes | 2 | 3 | 5 |
| Couches | 773 → **128** → 2 | 773 → **256** → **128** → 3 | 773 → **512** → **256** → 5 |
| Paramètres appris | 99 330 | 231 427 | 528 901 |
| Activation / init | ReLU / He, sortie Xavier | idem | idem |
| Normalisation | oui | oui | oui |
| Dropout | 0.2 | 0.2 | **0.3** |
| Loss | cross-entropy | cross-entropy | cross-entropy |
| Optimiseur | Adam, `lr=0.001`, `weight_decay=1e-4` | idem | idem |
| Époques max / patience | 30 / 5 | 40 / 5 | **50 / 7** |
| Batch / validation | 128 / 15 % | 128 / 15 % | 128 / 15 % |
| Graine | 42 | 42 | 42 |

**Logique d'ensemble** : plus la tâche distingue de classes, plus le réseau est **grand** (il doit apprendre plus de règles), plus il est **régularisé** (un grand réseau surapprend plus facilement) et plus on lui laisse de **temps** (plus de paramètres à ajuster).

---

## 4. Chaque hyperparamètre en détail

Pour chacun : rôle, effet d'une valeur trop basse / trop haute, valeur retenue et pourquoi.

### 4.1 Architecture : nombre et taille des couches

**Rôle** : la **capacité** du réseau, c'est-à-dire la complexité des règles qu'il peut représenter. Nombre de paramètres d'une couche `Linear` : $n_{in} \times n_{out} + n_{out}$.

| Trop petit | Trop grand |
| :--- | :--- |
| **Sous-apprentissage** : même la loss d'entraînement reste haute, le réseau ne peut pas représenter « roi attaqué + aucune case de fuite » | **Surapprentissage** (apprend par cœur), entraînement plus lent, fichiers `.nn` plus gros |

**Retenu** : 1 couche cachée pour `basic`, 2 pour `medium` et `full`, en **entonnoir** (chaque couche deux fois plus étroite que la précédente).

**Pourquoi** :
- L'entrée (773 features) décrit les pièces case par case. Détecter un échec demande de combiner la position d'une pièce attaquante, celle du roi et les cases entre les deux : au moins une couche cachée est nécessaire (une couche linéaire seule ne sait pas combiner, voir le XOR en [02 §2](02-theorie.md#2-pourquoi-des-activations-)), deux pour des règles imbriquées comme le mat.
- L'entonnoir compresse progressivement l'information vers les quelques classes de sortie.
- Des puissances de 2 (128, 256, 512) : convention, sans effet sur le résultat avec NumPy.

**À expérimenter** : 64 / 128 / 256 / 512 neurones, 1 à 3 couches cachées.

### 4.2 Fonction d'activation

| Choix | Avantage | Inconvénient |
| :--- | :--- | :--- |
| `relu` | Rapide, ne sature pas pour $x > 0$ | Neurones « morts » possibles (entrée toujours négative → plus jamais de gradient) |
| `leakyrelu` | Comme ReLU, sans neurones morts | Un hyperparamètre de plus (`alpha`) |
| `tanh`, `sigmoid` | Sorties bornées | **Saturent** : gradient ≈ 0 pour les grandes valeurs, apprentissage lent en profondeur |

**Retenu** : `relu`. C'est le standard pour les couches cachées, et nos réseaux sont peu profonds (le risque de neurones morts est faible, surtout avec la normalisation et Adam).

**À expérimenter** : `leakyrelu` (`alpha` 0.01 à 0.1) si beaucoup de neurones restent à 0 ; `tanh` pour la comparaison dans les benchmarks.

**`alpha` (LeakyReLU)** : pente pour $x < 0$. Trop petit (→ 0) : retour au comportement de ReLU. Trop grand (→ 1) : la fonction devient presque linéaire et le réseau perd sa capacité à combiner. Valeurs usuelles : 0.01 à 0.2.

### 4.3 Initialisation des poids

| Init | Variance | À utiliser avant |
| :--- | :--- | :--- |
| `he` | $2 / n_{in}$ | ReLU, LeakyReLU |
| `xavier` | $2 / (n_{in} + n_{out})$ | Tanh, Sigmoid, couche de sortie |
| `lecun` | $1 / n_{in}$ | SELU |

**Mauvais choix** : poids trop grands → les valeurs explosent de couche en couche (loss `nan`) ; trop petits → les signaux et les gradients s'éteignent, l'apprentissage stagne.

**Retenu** : `he` pour les couches cachées (suivies de ReLU, qui annule la moitié des valeurs : le facteur 2 compense), `xavier` pour la couche de sortie (pas suivie de ReLU, sorties équilibrées au départ). Justification théorique : [02 §7](02-theorie.md#7-initialisation-des-poids).

### 4.4 Normalisation des entrées

**Rôle** : ramener chaque feature à moyenne 0 et écart-type 1.

**Sans** : les features de grande échelle dominent les gradients. Mesuré (`tests/test_normalization.py`) : sur des features d'échelles ~1000 et ~0.001, SGD **diverge** (loss `nan`) sans normalisation, atteint **92 %** avec.

**Retenu** : présente dans les trois configs. Avec notre encodeur, les features valent 0 ou 1, mais elles sont très déséquilibrées (une case donnée contient rarement une dame, souvent un pion au départ) : la normalisation centre ces fréquences. Coût nul (aucun paramètre appris). Les features constantes sur l'entraînement sont seulement centrées (jamais de division par 0).

**À expérimenter** : avec / sans, pour le mesurer sur les vraies données.

### 4.5 Dropout

**Rôle** : éteindre au hasard une fraction `rate` des neurones à chaque pas d'entraînement, pour empêcher le réseau de dépendre de quelques neurones précis **[OPT1]**.

| Trop bas (→ 0) | Trop haut (→ 0.5 et plus) |
| :--- | :--- |
| Pas d'effet : surapprentissage possible | Le réseau perd trop d'information à chaque pas : apprentissage lent ou sous-apprentissage |

**Retenu** : 0.2 pour `basic` et `medium`, 0.3 pour `full` (le plus grand réseau, donc le plus exposé au surapprentissage). Valeurs usuelles : 0.1 à 0.5, 0.5 étant la valeur historique pour les très grandes couches.

**Où le placer** : après l'activation des couches cachées, jamais après la couche de sortie.

**`mc_dropout`** : `false` dans nos configs. À activer seulement pour estimer l'incertitude des prédictions ([09 §6](09-guide-utilisation.md#6-prédire)) ; en prédiction normale, il rendrait les sorties aléatoires.

### 4.6 Fonction de coût (`loss`)

| Loss | Adaptée à | Remarque |
| :--- | :--- | :--- |
| `cross_entropy` | Classification | Gradient $\hat{Y} - Y$ : fort quand le réseau se trompe beaucoup |
| `mse` | Régression | En classification, gradient faible quand le réseau se trompe avec assurance : apprentissage plus lent |

**Retenu** : `cross_entropy`, car notre tâche est une classification. Théorie : [02 §4](02-theorie.md#4-mesurer-lerreur--la-loss).

### 4.7 Optimiseur

| | `adam` | `sgd` (avec momentum) |
| :--- | :--- | :--- |
| Pas | Adapté **par poids** | Le même pour tous les poids |
| Sensibilité au `lr` | Faible : marche bien avec `0.001` dans la plupart des cas | Forte : il faut chercher le bon `lr` |
| Vitesse | Converge vite | Plus lent |
| Généralisation | Bonne | Parfois légèrement meilleure, une fois bien réglé |

**Retenu** : `adam`, pour converger vite sans réglage fin pendant le développement.

**À expérimenter** : `sgd` avec `momentum=0.9` et `lr` entre 0.01 et 0.1, pour la comparaison SGD / Adam prévue dans les benchmarks.

### 4.8 Learning rate (`lr`)

Le plus important des hyperparamètres d'entraînement.

| Trop bas | Trop haut |
| :--- | :--- |
| La loss baisse très lentement ; risque d'arriver à `epochs` avant d'avoir appris | La loss oscille, remonte ou devient `nan` (divergence) |

**Retenu** : `0.001`, la valeur par défaut recommandée par les auteurs d'Adam.

**À expérimenter** : échelle **logarithmique** : 0.0003, 0.001, 0.003, 0.01 (multiplier par ~3 à chaque essai, pas ajouter 0.001).

**Planning** (décroissance, warmup) : pas encore intégré à `fit()`. Possible à la main en modifiant `model.optimizer.lr` entre deux époques ([09 §8.3](09-guide-utilisation.md#83-boucle-dentraînement-manuelle-planning-de-learning-rate)).

### 4.9 Momentum (SGD) et $\beta_1$, $\beta_2$, $\epsilon$ (Adam)

| Hyperparamètre | Rôle | Valeur usuelle | Quand y toucher |
| :--- | :--- | :--- | :--- |
| `momentum` | Part de la vitesse précédente conservée | `0.9` | Presque jamais ; `0` pour comparer avec la descente de gradient simple |
| `beta1` | Mémoire de la moyenne des gradients | `0.9` | Presque jamais |
| `beta2` | Mémoire de la moyenne des carrés | `0.999` | Presque jamais |
| `eps` | Évite la division par 0 | `1e-8` | Jamais en pratique |

**Retenu** : valeurs par défaut, celles des publications originales. Les modifier apporte rarement quelque chose : mieux vaut régler `lr` d'abord.

### 4.10 Régularisation L2 (`weight_decay`)

**Rôle** : ajoute $\lambda \cdot w$ au gradient de chaque poids, ce qui revient à minimiser $L + \frac{\lambda}{2}\lVert W \rVert^2$ : les grands poids sont pénalisés, le réseau préfère des solutions plus « simples » **[OPT1]**.

| Trop bas (0) | Trop haut (≥ 0.01) |
| :--- | :--- |
| Pas de régularisation | Les poids sont tirés vers 0 plus fort que l'apprentissage ne les pousse : sous-apprentissage |

**Retenu** : `1e-4`, une régularisation légère, complémentaire du dropout.

**À expérimenter** : 0, 1e-5, 1e-4, 1e-3.

**Nuance à connaître** : avec Adam, cette pénalité passe par la division par $\sqrt{v}$, donc son effet varie d'un poids à l'autre (c'est la L2 « classique », pas AdamW). Elle s'applique aussi aux biais. Les deux choix sont volontairement simples ; l'effet reste faible avec $\lambda = 10^{-4}$.

### 4.11 Taille de batch (`batch_size`)

| Petit batch (8–32) | Grand batch (512 et plus) |
| :--- | :--- |
| Gradient bruité : plus de pas par époque, le bruit peut aider à sortir des mauvais minima, mais l'entraînement est plus lent avec NumPy | Gradient précis et calcul matriciel efficace, mais moins de pas par époque et parfois moins bonne généralisation |

**Retenu** : `128`, un compromis courant : assez grand pour que NumPy calcule efficacement, assez petit pour faire beaucoup de pas par époque.

**Interaction** : si on multiplie `batch_size` par 4, il faut souvent augmenter `lr` (un peu) pour garder la même vitesse d'apprentissage. Changer les deux en même temps rend les comparaisons difficiles : en changer un à la fois.

### 4.12 Époques (`epochs`) et patience (`patience`)

- **`epochs`** est une **limite haute**. Grâce à l'early stopping, l'entraînement s'arrête souvent avant.
- **`patience`** : nombre d'époques sans amélioration de `val_loss` avant l'arrêt. Les meilleurs poids sont ensuite restaurés.

| | Trop bas | Trop haut |
| :--- | :--- | :--- |
| `epochs` | Arrêt avant convergence (la `val_loss` baissait encore) | Aucun risque grâce à l'early stopping, seulement du temps perdu si la patience est grande |
| `patience` | Arrêt sur une simple fluctuation de la validation | Beaucoup d'époques inutiles après le meilleur point (sans danger : les meilleurs poids sont restaurés) |

**Retenu** : 30 / 40 / 50 époques et une patience de 5 (7 pour `full`) : les grands réseaux progressent plus lentement et leur `val_loss` fluctue davantage.

**Vérifier** : si `history["best_epoch"]` est égal au nombre d'époques, l'entraînement n'avait pas fini : augmenter `epochs`.

### 4.13 Part de validation (`val_split`)

**Rôle** : fraction des données mise de côté pour mesurer la généralisation et piloter l'early stopping.

| Trop petite (< 5 %) | Trop grande (> 30 %) |
| :--- | :--- |
| Mesure bruitée, peu d'exemples des classes rares (les mats) : early stopping peu fiable | Moins de données pour apprendre |

**Retenu** : `0.15`, valeur usuelle (souvent avec 15 % de test en plus, voir [09 §4.2](09-guide-utilisation.md#42-séparer-entraînement--validation--test)).

**Limite actuelle** : le découpage est aléatoire, pas **stratifié** (il ne garantit pas la même proportion de chaque classe dans la validation). Avec un grand jeu de données, l'écart est faible.

### 4.14 Graine (`seed`)

**Rôle** : fixe le hasard (poids initiaux, mélange, découpage de validation, masques de dropout) pour que l'entraînement soit **reproductible**, exigence du sujet.

**Retenu** : `42` (valeur arbitraire, seule compte qu'elle soit fixée). Le générateur utilise `seed + k - 1` pour le *k*-ième réseau.

**Bonne pratique** : une seule graine peut avoir de la chance. Pour comparer deux réglages de façon fiable, entraîner chacun avec 3 graines et comparer les moyennes.

---

## 5. Méthode de réglage

### 5.1 Dans quel ordre ?

Changer **un seul** hyperparamètre à la fois, et toujours juger sur la **validation** (jamais sur le test).

1. **Vérifier que le réseau peut apprendre** : sur un petit sous-ensemble (quelques centaines de positions), sans dropout ni L2, la loss d'entraînement doit tomber près de 0. Sinon, il y a un problème de données ou de code, pas d'hyperparamètres.
2. **Learning rate** : tester 0.0003 / 0.001 / 0.003 / 0.01 avec la config de base. Garder le meilleur.
3. **Capacité** (taille et nombre de couches) : augmenter tant que la `val_loss` s'améliore.
4. **Régularisation** (dropout, `weight_decay`) : l'augmenter si `train_loss` ≪ `val_loss` (surapprentissage).
5. **Taille de batch, optimiseur** : en dernier, ils changent moins le résultat.
6. **Confirmer** le réglage final avec plusieurs graines, puis mesurer **une seule fois** sur le test.

### 5.2 Diagnostic : symptôme → réglage

| Ce qu'on observe | Probable cause | Réglage |
| :--- | :--- | :--- |
| Loss `nan` ou qui explose | `lr` trop grand | Diviser `lr` par 3 à 10 ; vérifier la normalisation |
| Loss qui baisse à peine | `lr` trop petit, ou réseau trop petit | Augmenter `lr`, puis la taille des couches |
| `train_loss` basse, `val_loss` qui remonte | Surapprentissage | Plus de dropout ou de `weight_decay`, réseau plus petit, plus de données |
| `train_loss` et `val_loss` hautes et proches | Sous-apprentissage | Réseau plus grand, plus d'époques, moins de régularisation |
| `val_loss` très irrégulière | Validation trop petite, ou `lr` trop grand | Augmenter `val_split`, baisser `lr`, augmenter `patience` |
| Arrêt dès les premières époques | `patience` trop faible, ou `lr` trop grand | Augmenter `patience`, baisser `lr` |
| `best_epoch` = dernière époque | L'entraînement n'était pas fini | Augmenter `epochs` |
| Bonne accuracy, mais les mats jamais trouvés | Classes déséquilibrées | Rééquilibrer le jeu de données ; regarder le rappel par classe ([09 §5](09-guide-utilisation.md#5-évaluer)) |

### 5.3 Recherche automatique [OPT2]

Le sujet demande d'aller au-delà du réglage manuel :

| Méthode | Principe | Coût |
| :--- | :--- | :--- |
| **Grid search** | Toutes les combinaisons d'une grille de valeurs | Explose avec le nombre d'hyperparamètres (4 valeurs × 4 hyperparamètres = 256 entraînements) |
| **Random search** | Combinaisons tirées au hasard dans des intervalles | Même budget, souvent meilleur : explore plus de valeurs de chaque hyperparamètre |
| **Algorithme génétique** | Population de configs ; on garde les meilleures, on les croise et on les mute | Plus complexe, efficace pour de grands espaces |
| **Recuit simulé** | On modifie une config au hasard, on accepte parfois une moins bonne pour sortir des minima locaux | Simple, séquentiel |

Une grid search écrite à la main est donnée dans le [guide §8.1](09-guide-utilisation.md#81-comparer-plusieurs-configurations-recherche-dhyperparamètres). Un outil intégré à la librairie reste à écrire.

**Espace de recherche proposé** pour commencer :

| Hyperparamètre | Valeurs à tester |
| :--- | :--- |
| `lr` | 0.0003, 0.001, 0.003, 0.01 |
| Taille des couches | 128, 256, 512 (première couche cachée) |
| Nombre de couches cachées | 1, 2, 3 |
| `dropout.rate` | 0.0, 0.2, 0.3, 0.5 |
| `weight_decay` | 0, 1e-5, 1e-4, 1e-3 |
| `batch_size` | 64, 128, 256 |
| `optimizer.type` | adam, sgd (`lr` × 10 pour SGD) |

---

## 6. Ce qui n'est pas (encore) réglable

| Hyperparamètre | Situation |
| :--- | :--- |
| Planning de learning rate (decay, warmup) | Pas dans `fit()` ; possible avec une boucle manuelle |
| Poids des classes dans la loss | Absent : les classes rares pèsent peu |
| Découpage de validation stratifié | Absent |
| Métrique surveillée par l'early stopping | Toujours `val_loss` (pas `val_acc`) ; pas de seuil minimal d'amélioration |
| Gradient clipping | Absent |
| Activation différente pour la sortie | La sortie est toujours des logits passés à la softmax de la loss |

---

## 7. Résultats expérimentaux (à compléter)

Section à remplir avec les mesures de la recherche d'hyperparamètres et des benchmarks (issue #6). Ce sont **ces résultats** qui justifieront les valeurs finales en soutenance.

**Protocole** (à respecter pour que les comparaisons soient valables) :
- même jeu de données, même découpage entraînement / validation / test pour tous les essais ;
- une seule valeur modifiée par rapport à la config de référence ;
- 3 graines par essai, on reporte la moyenne ;
- choix sur la **validation**, score final sur le **test** une seule fois.

**Jeu de données utilisé** : _à compléter (nombre de positions, répartition des classes, script de génération)._

| Config de référence | Hyperparamètre modifié | Valeur | `val_loss` (moy.) | `val_acc` (moy.) | Meilleure époque | Remarque |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `medium_network.conf` | — (référence) | — | | | | |
| `medium_network.conf` | `lr` | 0.0003 | | | | |
| `medium_network.conf` | `lr` | 0.003 | | | | |
| `medium_network.conf` | `dropout.rate` | 0.0 | | | | |
| `medium_network.conf` | `dropout.rate` | 0.5 | | | | |
| `medium_network.conf` | `weight_decay` | 0 | | | | |
| `medium_network.conf` | `optimizer` | sgd, `lr=0.01` | | | | |

**Valeurs finales retenues** : _à compléter, avec pour chacune la ligne du tableau qui la justifie._

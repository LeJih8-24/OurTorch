# 08 — Préparer la soutenance

Questions probables, réponses courtes, et où trouver la preuve dans le code. Les réponses renvoient aux chapitres détaillés.

## Contraintes du sujet

**« Avez-vous utilisé une librairie de réseaux de neurones ? »**
Non. Seul NumPy sert au calcul matriciel (`@`, `exp`, `sum`…). Le forward, la rétropropagation, les losses et les optimiseurs sont écrits à la main dans `mytorch/` : chaque `backward()` est visible dans `mytorch/layers/`, et `optimizers.py` contient les formules de SGD et Adam.

**« Comment prouvez-vous que vos gradients sont justes ? »**
Par le **gradient check** : chaque couche et chaque loss est comparée à une dérivée numérique par différences finies, avec une erreur relative < $10^{-5}$ (`tests/gradcheck.py`). Voir [07](07-tests.md#le-gradient-check).

**« Le prédicteur utilise-t-il un algorithme d'échecs ? »**
Non. Un algorithme d'échecs peut seulement servir à **générer les labels** du jeu d'entraînement, ce que le sujet autorise. La prédiction vient uniquement du réseau entraîné. L'encodeur (`analyzer/fen_encoder.py`) ne donne au réseau que la **description brute** de la position (pièces par case, trait, roques) : aucune feature ne calcule « le roi est attaqué ». C'est le réseau qui doit apprendre cette règle.

**« Où est la gestion d'erreurs ? »**
Toutes les erreurs passent par `exit_error()` (`analyzer/utils.py`) : message sur stderr, code 84. La librairie lève des `ValueError` aux messages précis, transformées en `exit_error` par les exécutables. Voir [06](06-analyseur-et-outils.md).

## Architecture du réseau

**« Pourquoi ReLU ? »**
Pas de saturation pour $x > 0$ (le gradient vaut 1, il ne disparaît pas dans un réseau profond), et très rapide à calculer. Sigmoid et Tanh saturent pour les grandes valeurs. LeakyReLU est disponible contre les neurones morts.

**« Pourquoi pas de softmax en dernière couche ? »**
Parce que `SoftmaxCrossEntropyLoss` l'applique déjà, sur les logits bruts. Combinées, softmax et cross-entropy ont un gradient très simple, $\hat{Y} - Y$, et numériquement stable. Ajouter une `Softmax` l'appliquerait deux fois ; la validation des `.conf` l'interdit. Voir [02 §6](02-theorie.md#le-cas-élégant-softmax--cross-entropy).

**« Pourquoi He pour les couches cachées et Xavier pour la sortie ? »**
He ($\text{Var} = 2/n_{in}$) compense le fait que ReLU annule la moitié des valeurs : l'amplitude du signal reste stable de couche en couche. La couche de sortie n'est pas suivie de ReLU, donc Xavier est plus adapté. Voir [02 §7](02-theorie.md#7-initialisation-des-poids).

**« Pourquoi des réseaux plus grands pour plus de classes ? »**
Distinguer 5 classes (avec la couleur du camp) demande d'apprendre plus de règles que 2. On augmente la capacité (512 → 256 neurones) et la régularisation (dropout 0.3).

**« Combien de paramètres ? »**
$n_{in} \times n_{out} + n_{out}$ par couche `Linear`. Pour `medium_network.conf` : $(773 \cdot 256 + 256) + (256 \cdot 128 + 128) + (128 \cdot 3 + 3) = 198\,144 + 32\,896 + 387 = 231\,427$ (vérifiable avec `sum(p.size for p, _ in model.parameters())`).

## Entraînement

**« Pourquoi Adam ? »**
Il adapte le pas de chaque poids en divisant par la moyenne des carrés de ses gradients, et sa correction de biais le rend efficace dès les premiers pas. Il converge vite sans réglage fin du learning rate. SGD + momentum est disponible pour comparer. Voir [04 §2](04-entrainement.md#2-les-optimiseurs-optimizerspy).

**« Comment évitez-vous l'overfitting ? » [OPT1]**
Trois mécanismes : (1) **séparation entraînement / validation** et **early stopping** avec restauration des meilleurs poids, (2) **Dropout**, (3) **régularisation L2** (`weight_decay`). On le démontre avec les courbes `train_loss` / `val_loss` du `history` : sans régularisation, la validation remonte ; avec, elle suit l'entraînement. (Graphiques à produire dans #6.)

**« Pourquoi normaliser les entrées ? »**
Pour que toutes les features aient la même échelle : sinon les grandes features dominent les gradients. Démontré par un test : SGD diverge (NaN) sans normalisation et atteint 92 % avec. Les statistiques sont calculées **sur l'entraînement uniquement** (pas de fuite d'information depuis la validation) et sauvegardées dans le `.nn`. Voir [03 — Normalization](03-couches.md#normalization-normalizationpy).

**« Pourquoi des mini-batchs ? » [OPT3]**
Un compromis : plus rapide et plus stable qu'un exemple à la fois, moins coûteux en mémoire que tout le jeu d'un coup. Le bruit des mini-batchs aide aussi à sortir des mauvais minima locaux.

**« L'entraînement est-il reproductible ? »**
Oui. La graine du `.conf` fixe l'initialisation ; le mélange des données, le découpage de validation et le dropout utilisent le même générateur aléatoire. Un test vérifie que deux entraînements avec la même graine donnent exactement les mêmes loss.

## Formats de fichiers

**« Que contient un `.nn` et pourquoi ce format ? »**
Une archive `.npz` : un en-tête versionné, la config en JSON et les tableaux de poids et de normalisation. Choisi plutôt que `pickle` pour la **sécurité** (chargé avec `allow_pickle=False`, il ne peut pas exécuter de code), la **robustesse** (indépendant des noms de classes Python) et la **lisibilité**. Voir [05 §2](05-configuration-et-sauvegarde.md#2-les-fichiers-nn-serializationpy).

**« Que se passe-t-il si l'entraînement plante pendant la sauvegarde ? »**
L'écriture est atomique : fichier temporaire, puis renommage. L'ancien `.nn` reste intact.

**« Pourquoi JSON pour les `.conf` ? »**
Lisible, modifiable à la main, sans dépendance, et identique à la sortie de `get_config()`. La validation stricte nomme le champ fautif en cas d'erreur.

**« Pourquoi les réseaux générés sont-ils différents si la graine est fixée ? »**
Le réseau *k* reçoit la graine `seed + k - 1` : les réseaux diffèrent entre eux, mais la même commande redonne toujours les mêmes fichiers.

## Conformité de `mytorch` au sujet

État au 28/10/2026, exigence par exigence (`SPECIFICATIONS.MD`) :

| Exigence | Niveau | État | Où |
| :--- | :--- | :--- | :--- |
| Aucune librairie de réseaux de neurones, uniquement NumPy | DOIT | ✅ | Tout `mytorch/` |
| Module importable | DOIT | ✅ | `import mytorch` |
| [NN1] Créer un réseau | DOIT | ✅ | `build_from_config`, `Sequential` |
| [NN2] Sauvegarder / charger | DOIT | ✅ | `save_network`, `load_network` |
| [NN3] Entraîner | DOIT | ✅ | `Sequential.fit` |
| Hyperparamètres : topologie, activations par couche, loss, taille de batch, époques | Liste indicative | ✅ | `.conf` |
| Hyperparamètres : optimiseur (SGD, momentum, Adam), L2, dropout, initialisation, graine | Liste indicative | ✅ | `.conf` |
| Hyperparamètres : **planning de learning rate** (decay, warmup) | Liste indicative | ❌ | Faisable à la main (`optimizer.lr`, guide 09 §8.3) |
| [GEN1] Générateur depuis un fichier de configuration | Conseillé | ✅ | `my_torch_generator` |
| [GEN2] Génération dans le code | Conseillé | ✅ | `Sequential(...)` |
| Format de fichier justifié (topologie, poids, normalisation, encodage des sorties) | DEVRAIT | ✅ | [05](05-configuration-et-sauvegarde.md) |
| Justification des hyperparamètres | DEVRAIT | ⚠️ | [10](10-hyperparametres.md) : justifications de principe écrites, mesures à compléter (§7) |
| [OPT1] Lutte contre l'overfitting | **DOIT** | ✅ | Validation, early stopping, dropout, L2 |
| [OPT1] Jeu de test, validation croisée | Exemple cité | ⚠️ | Test : à la main (guide 09 §4.2) ; validation croisée absente |
| [OPT2] Optimisation des hyperparamètres (grid / random search, génétique…) | Attendu en soutenance | ❌ | Grid search écrite à la main dans le guide (09 §8.1), pas d'outil dans la librairie |
| [OPT3] Mini-batch, initialisations | Exemples cités | ✅ | `fit`, `initializers.py` |
| [OPT3] MC Dropout | Exemple cité | ⚠️ | Couche prête, pas de méthode de prédiction dédiée |
| Benchmarks avec graphiques | DEVRAIT | ❌ | `history` fournit les données ; script et graphiques : issue #6 |
| [B2] Plusieurs courbes sur un graphique | Bonus | ❌ | Issue #6 |
| [B3] Métriques multiples (précision, rappel, F1) | Bonus | ❌ | Calcul à la main dans le guide (09 §5) |
| [B1] Parallélisme | Bonus | ❌ | — |

**Bilan** : toutes les exigences **obligatoires** de la librairie sont remplies. Ce qui manque relève des attentes de soutenance ([OPT2], benchmarks) et des bonus.

## Limites connues et améliorations prévues

Il vaut mieux les connaître et les annoncer avant qu'on les relève :

| Limite | Impact | Piste |
| :--- | :--- | :--- |
| Pas de **pondération des classes** dans la cross-entropy | `Checkmate` est rare : le réseau peut négliger cette classe | Poids par classe dans la loss |
| Découpage de validation **non stratifié** | La validation peut contenir très peu de mats | Découpage qui respecte les proportions de chaque classe |
| Seule l'**accuracy** est mesurée | Elle masque les erreurs sur les classes rares | Matrice de confusion, précision, rappel, F1 par classe [B3] |
| Pas de **planning de learning rate** | Réglage manuel du `lr` | Décroissance par paliers ou cosinus (`optimizer.lr` est déjà modifiable) |
| MC Dropout non exploité côté modèle | Pas d'estimation d'incertitude | `Sequential.predict_mc(X, T)` |
| Pas de **recherche d'hyperparamètres** [OPT2] | Attendue en soutenance | Grid search / random search / algorithme génétique dans la librairie |
| Pas de **validation croisée** | Mesure de performance moins fiable sur un petit jeu de données | k-fold : entraîner k fois sur k découpages et moyenner |
| État de l'optimiseur non sauvegardé | Un réentraînement repart avec Adam « à froid » | Sauvegarder `m`, `v`, `t` si besoin |
| Le `weight_decay` s'applique aussi aux biais | Effet négligeable en pratique | Exclure les biais |

## Démos possibles en soutenance

```bash
# 1. Tous les tests, dont les gradient checks
.venv/bin/pytest tests/ -v -k gradient

# 2. Générer des réseaux reproductibles (NumPy doit être disponible pour python3)
source .venv/bin/activate
make re && ./my_torch_generator configs/basic_network.conf 2

# 3. Une erreur de configuration claire, code 84
./my_torch_generator configs/does_not_exist.conf 1; echo "code: $?"
```

```python
# 4. Le réseau apprend le XOR (non linéairement séparable)
import numpy as np
from mytorch import Adam, Sequential, SoftmaxCrossEntropyLoss
from mytorch.layers import Linear, ReLU

np.random.seed(42)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=np.float32)
model = Sequential([Linear(2, 8), ReLU(), Linear(8, 2)], SoftmaxCrossEntropyLoss(), Adam(lr=0.05), ["0", "1"])
model.fit(X, y, epochs=300, batch_size=4, val_split=0.0)
print(model.predict(X))   # ['0', '1', '1', '0']
```

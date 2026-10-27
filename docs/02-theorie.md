# 02 — Théorie : comment un réseau apprend

Ce chapitre explique les notions nécessaires pour comprendre (et défendre) le code. Chaque notion renvoie au fichier qui l'implémente.

## 1. Un neurone, une couche, une matrice

Un **neurone** fait une somme pondérée de ses entrées, plus un biais :

$$y = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$$

Une **couche** de `out` neurones qui reçoivent chacun les mêmes `in` entrées s'écrit avec une matrice. Pour un **batch** de $N$ positions :

$$Y = X W + b \qquad X : (N, \text{in}),\ W : (\text{in}, \text{out}),\ b : (1, \text{out}),\ Y : (N, \text{out})$$

C'est exactement `Linear.forward()` ([dense.py](../mytorch/layers/dense.py)) : `inputs @ self.W + self.b`. Le biais `b` de forme `(1, out)` est automatiquement ajouté à chaque ligne grâce au *broadcasting* de NumPy.

**Pourquoi des batchs ?** Une multiplication de matrices sur 128 positions est beaucoup plus rapide que 128 multiplications séparées (NumPy utilise des routines optimisées), et le gradient moyen sur un batch est moins bruité que celui d'une seule position.

## 2. Pourquoi des activations ?

Empiler des couches `Linear` sans rien entre elles ne sert à rien : $X W_1 W_2 = X (W_1 W_2)$, c'est encore une seule couche linéaire. Un réseau linéaire ne peut tracer que des frontières droites entre les classes.

Une **fonction d'activation non linéaire** entre deux couches (ex. ReLU : $\max(0, x)$) permet au réseau d'apprendre des frontières courbes, donc des règles complexes (« le roi est attaqué ET n'a aucune case de fuite »).

Exemple classique, testé dans `tests/test_xor.py` : le **XOR** (`0⊕0=0, 0⊕1=1, 1⊕0=1, 1⊕1=0`) n'est pas séparable par une droite. `Linear(2, 8) → ReLU → Linear(8, 2)` le résout à 100 %.

Activations disponibles : [03 — Les couches](03-couches.md#activations-activationspy).

## 3. De la sortie à une décision : softmax

La dernière couche produit un nombre par classe, les **logits** (ex. `[-1.2, 3.4, 0.7]`). La **softmax** les transforme en probabilités positives qui somment à 1 :

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

`[-1.2, 3.4, 0.7]` → `[0.01, 0.93, 0.06]`. La classe prédite est celle de plus grande probabilité (`argmax`).

**Astuce de stabilité** : $e^{1000}$ dépasse la capacité d'un nombre flottant. On soustrait d'abord le maximum ($e^{z_i - \max z}$), ce qui ne change pas le résultat (le facteur $e^{-\max z}$ se simplifie) mais garde des valeurs ≤ 1. Implémenté dans `softmax()` ([losses.py](../mytorch/losses.py)) et `Softmax` ([activations.py](../mytorch/layers/activations.py)).

## 4. Mesurer l'erreur : la loss

La **loss** est un nombre qui dit à quel point le réseau se trompe. L'entraînement consiste à la **minimiser**.

Pour la classification, on utilise l'**entropie croisée** (cross-entropy). Avec $Y$ les labels one-hot et $\hat{Y}$ les probabilités prédites :

$$L = -\frac{1}{N} \sum_{n} \sum_{i} Y_{n,i} \log \hat{Y}_{n,i}$$

Comme $Y$ est one-hot, seule la probabilité de la **bonne** classe compte : $L = -\log(\text{proba de la bonne réponse})$.
- Proba 0.93 sur la bonne classe → $L = 0.07$ (bien).
- Proba 0.01 sur la bonne classe → $L = 4.6$ (très mauvais, fortement pénalisé).

Pour 3 classes, un réseau qui répond au hasard (1/3 partout) a une loss de $\log 3 \approx 1.10$ : c'est la valeur de départ typique.

Détails et MSE : [04 — Entraînement](04-entrainement.md#1-les-losses-lossespy).

## 5. Descente de gradient

Le **gradient** $\frac{\partial L}{\partial w}$ indique comment la loss varie quand on augmente un peu le poids $w$ :
- gradient positif → augmenter $w$ augmente l'erreur → il faut **diminuer** $w$ ;
- gradient négatif → il faut **augmenter** $w$.

D'où la règle de base (SGD), répétée des milliers de fois :

$$w \leftarrow w - \eta \cdot \frac{\partial L}{\partial w}$$

$\eta$ est le **learning rate** (taux d'apprentissage) : trop grand, l'entraînement diverge ; trop petit, il est très lent. Les optimiseurs (`SGD` avec momentum, `Adam`) sont des variantes plus efficaces de cette règle : [04 — Entraînement](04-entrainement.md#2-les-optimiseurs-optimizerspy).

« Stochastique » / « mini-batch » : le gradient est calculé sur un petit paquet de positions tiré au hasard, pas sur tout le jeu de données. C'est plus rapide, et le bruit aide à sortir des mauvais minima locaux ([OPT3]).

## 6. La rétropropagation (backpropagation)

Un réseau a des centaines de milliers de poids. Calculer chaque gradient séparément serait beaucoup trop lent. La **rétropropagation** les calcule tous en **un seul passage à l'envers**, grâce à la **règle de la chaîne** :

$$\text{si } L = f(g(x)) \text{ alors } \frac{\partial L}{\partial x} = \frac{\partial L}{\partial g} \cdot \frac{\partial g}{\partial x}$$

Concrètement, chaque couche n'a besoin que d'une chose : le gradient de la loss **par rapport à sa sortie** $G = \frac{\partial L}{\partial Y}$, reçu de la couche suivante. Elle en déduit :
1. le gradient par rapport à **ses poids** (pour l'optimiseur) ;
2. le gradient par rapport à **son entrée** $\frac{\partial L}{\partial X}$, qu'elle renvoie à la couche précédente.

```
forward :   X ──Linear──► H ──ReLU──► A ──Linear──► Z ──loss──► L
backward:  dX ◄─────────  dH ◄──────  dA ◄─────────  dZ ◄──────── 1
                 ↓ dW1                       ↓ dW2
```

C'est le rôle de `backward(grad_output)` dans chaque couche, et de `Sequential.backward()` qui appelle les couches dans l'ordre inverse ([model.py](../mytorch/model.py)).

### Exemple : les gradients de `Linear`

Pour $Y = XW + b$, avec $G = \frac{\partial L}{\partial Y}$ de forme $(N, \text{out})$ :

| Gradient | Formule | Forme | Code |
| :--- | :--- | :--- | :--- |
| Poids | $\frac{\partial L}{\partial W} = X^\top G$ | `(in, out)`, comme `W` | `self.dW[...] = self._cache_X.T @ grad_output` |
| Biais | $\frac{\partial L}{\partial b} = \sum_{\text{lignes}} G$ | `(1, out)`, comme `b` | `self.db[...] = np.sum(grad_output, axis=0, keepdims=True)` |
| Entrée | $\frac{\partial L}{\partial X} = G W^\top$ | `(N, in)`, comme `X` | `return grad_output @ self.W.T` |

**Astuce pour retrouver ces formules** : le gradient d'un tableau a toujours **la même forme** que ce tableau. Il n'y a qu'une façon de combiner $X$ (`N×in`) et $G$ (`N×out`) pour obtenir `in×out` : $X^\top G$.

**Pourquoi `forward()` garde $X$ en mémoire (`_cache_X`) ?** Parce que $\frac{\partial L}{\partial W}$ dépend de $X$. Chaque couche mémorise pendant le forward ce dont elle aura besoin au backward.

### Le cas élégant softmax + cross-entropy

Dériver softmax puis cross-entropy séparément est compliqué. Combinées, la dérivée par rapport aux logits se simplifie énormément :

$$\frac{\partial L}{\partial Z} = \frac{\hat{Y} - Y}{N}$$

« prédiction moins vérité », moyenné sur le batch. C'est pour cela que `SoftmaxCrossEntropyLoss` applique elle-même la softmax et que le réseau se termine par des logits bruts, **sans** couche `Softmax`.

### Comment être sûr que les gradients sont justes ?

Avec le **gradient check** : on compare le gradient calculé par `backward()` à une approximation numérique

$$\frac{\partial L}{\partial x} \approx \frac{L(x + \varepsilon) - L(x - \varepsilon)}{2\varepsilon}$$

Si l'écart relatif est < $10^{-5}$, le `backward()` est correct. Chaque couche et chaque loss est vérifiée ainsi : [07 — Tests](07-tests.md#le-gradient-check).

## 7. Initialisation des poids

Si tous les poids valent 0, tous les neurones d'une couche calculent la même chose et reçoivent le même gradient : ils restent identiques pour toujours. Il faut donc des poids **aléatoires**.

Mais leur **échelle** compte : trop grands, les valeurs explosent de couche en couche ; trop petits, elles s'effondrent vers 0 et les gradients disparaissent. Les stratégies classiques choisissent la variance pour que l'amplitude des signaux reste stable :

| Stratégie | Variance | Adaptée à |
| :--- | :--- | :--- |
| **He** (Kaiming) | $\frac{2}{n_{in}}$ | ReLU (qui annule la moitié des valeurs, d'où le facteur 2) |
| **Xavier** (Glorot) | $\frac{2}{n_{in} + n_{out}}$ | Tanh, Sigmoid, couche de sortie |
| **LeCun** | $\frac{1}{n_{in}}$ | SELU / réseaux auto-normalisés |

Implémenté dans [initializers.py](../mytorch/initializers.py). Nos configs utilisent He pour les couches cachées (suivies de ReLU) et Xavier pour la couche de sortie.

## 8. Normalisation des entrées

Si une feature varie entre 0 et 1000 et une autre entre 0 et 0.001, les poids liés à la première reçoivent des gradients énormes et l'entraînement devient instable. **Standardiser** chaque feature (moyenne 0, écart-type 1) met toutes les features à égalité.

Mesuré dans `tests/test_normalization.py` : sur des données à ces échelles, SGD **diverge** (loss = NaN) sans normalisation et atteint **92 %** de précision avec. Implémentation : [03 — Les couches](03-couches.md#normalization-normalizationpy).

## 9. Le surapprentissage et comment le combattre ([OPT1])

Un réseau assez grand peut **apprendre par cœur** le jeu d'entraînement : erreur quasi nulle sur les positions vues, mauvaise sur les nouvelles. C'est l'**overfitting**.

On le **détecte** en séparant les données :
- **entraînement** : sert à calculer les gradients ;
- **validation** : jamais utilisée pour apprendre, mesure la généralisation à chaque époque ;
- **test** : utilisé une seule fois à la fin, pour le score final honnête.

Quand la loss d'entraînement baisse mais que celle de validation remonte, le réseau surapprend.

On le **combat** avec trois mécanismes déjà implémentés :

| Technique | Principe | Où |
| :--- | :--- | :--- |
| **Early stopping** | On arrête quand la validation ne s'améliore plus depuis `patience` époques, et on restaure les meilleurs poids | `Sequential.fit()` |
| **Dropout** | À l'entraînement, on éteint des neurones au hasard : le réseau ne peut pas dépendre d'un neurone précis | `Dropout` |
| **Régularisation L2** (weight decay) | On pénalise les grands poids : $L + \frac{\lambda}{2}\lVert W\rVert^2$, ce qui favorise des solutions plus simples | `weight_decay` des optimiseurs |

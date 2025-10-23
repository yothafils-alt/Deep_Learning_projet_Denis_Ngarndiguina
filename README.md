# Deep_Learning_projet_Denis_Ngarndiguina
Comparaison de classification chien et chat d'un modèle entraîné à partir de zéro et d'un modèle utilisé avec le transfert d'apprentissage
# Étude comparative : CNN à partir de zéro vs transfert d'apprentissage pour la classification chats vs chiens
## Objectif
Ce projet vise à réaliser une étude comparative entre l'entraînement d'un réseau neuronal convolutif (CNN) à partir de zéro et l'utilisation d'une approche d'apprentissage par transfert pour la tâche de classification d'images en « chat » ou « chien ». L'étude évalue les performances des deux méthodes à l'aide de mesures clés telles que la précision, l'exactitude et le rappel, et analyse leur comportement de convergence ainsi que l'impact des techniques de régularisation et des stratégies d'optimisation.

## Environnement
Pour exécuter ce projet, vous devez installer les bibliothèques suivantes :
- `Python 3.x`
- `torch`
- `torchvision`
- `torchmetrics`
- `h5py`
- `numpy`
- `matplotlib`
- `Pillow`

Un GPU est fortement recommandé pour accélérer l'entraînement, en particulier pour l'expérience d'apprentissage par transfert. Le code inclut une vérification de la disponibilité du GPU et l'utilisera s'il est détecté.

## Données
L'ensemble de données utilisé pour ce projet est un sous-ensemble de l'ensemble de données populaire Cats vs Dogs.
- **Source :** ensemble de données Kaggle Cats and Dogs (un sous-ensemble est utilisé).
- **Format :** l'ensemble de données est fourni au format HDF5.
- **Fichiers :** `trainset.hdf5` et `testset.hdf5`.
- **Contenu :** chaque fichier HDF5 contient des données d'images (X_train/X_test) sous forme de tableaux NumPy de forme (N, 64, 64) avec le type de données `uint8`, et les étiquettes correspondantes (Y_train/Y_test) sous forme de tableaux NumPy de forme (N, 1) avec le type de données `float64`.
- **Taille :**
- Ensemble d'entraînement : 1 000 images
- Ensemble de test : 200 images

Les images sont en niveaux de gris (64x64) et sont transformées en images RVB 224x224 à l'aide de `torchvision.transforms` pour assurer la compatibilité avec les modèles pré-entraînés et appliquer l'augmentation des données.

## Commandes
Pour reproduire l'étude, exécutez les cellules de code de manière séquentielle dans le notebook Jupyter fourni. Les principales étapes sont les suivantes :

1.  **Configuration de l'environnement :** montez Google Drive (si vous utilisez Colab), vérifiez la disponibilité du GPU et définissez une graine aléatoire pour la reproductibilité.
2.  **Chargement et transformation des données :** chargez les ensembles de données HDF5 et appliquez des transformations d'images (y compris le redimensionnement, l'augmentation des données pour l'entraînement et la normalisation).
3.  **CNN à partir de zéro (expérience A) :**
- Définissez l'architecture `SimpleCNN` avec des couches convolutives, la normalisation par lots et le dropout.
- Entraînez le modèle CNN à l'aide des données d'entraînement. La boucle d'entraînement suit la perte, l'exactitude, la précision et le rappel pour les ensembles d'entraînement et de validation.
- 4.  **Apprentissage par transfert (expérience B) :**

- Chargez un modèle ResNet50 pré-entraîné à partir de `torchvision.models`.

- Gelez les couches pré-entraînées et remplacez/affinez la couche finale entièrement connectée pour la classification binaire.

    - Entraînez le modèle d'apprentissage par transfert. Cela implique d'expérimenter différents optimiseurs (Adam, SGD) et différents taux d'apprentissage (e.001, 0,01 pour SGD). La boucle d'entraînement suit également les indicateurs clés.

- Enregistrez le modèle d'apprentissage par transfert le plus performant.

5.  **Évaluation :**

- Chargez les meilleurs modèles enregistrés (ou utilisez les modèles entraînés s'ils ne sont pas rechargés).

    - Évaluer les performances finales des deux modèles sur l'ensemble de test à l'aide de l'exactitude, de la précision et du rappel.

- Générer une matrice de confusion pour le modèle le plus performant.

6.  **Visualisation :**

- Tracer les courbes métriques (perte, exactitude, précision, rappel) sur plusieurs époques pour les deux expériences (CNN à partir de zéro et apprentissage par transfert) afin de visualiser la convergence.

    - Affichez le graphique de la matrice de confusion.

7.  **Comparaison et analyse :**

- Comparez les mesures finales de l'ensemble de test.

- Analysez la matrice de confusion.

- Discutez du comportement de convergence, de l'impact de la régularisation et de l'optimisation, et fournissez une comparaison complète des deux approches.

## Résultats

### Indicateurs de performance de l'ensemble de tests finaux

| Modèle                     | Précision | Précision | Rappel |
|---------------------------|----------|-----------|--------|
| CNN à partir de zéro          | 0,5000   | 0,0000    | 0,0000 |
| Apprentissage par transfert (Adam, lr=0,001) | 0,8350   | 0,7965    | 0,9000 |
| Apprentissage par transfert (SGD, lr=0,001) | 0,8400   | 0,8000    | 0,9000 |
| Apprentissage par transfert (SGD, lr=0,01) | 0,8350   | 0,7815    | 0,9300 |

**Remarque :** le modèle CNN à partir de zéro n'a pas réussi à apprendre efficacement la tâche, ce qui a donné lieu à des performances équivalentes au hasard (précision ~50 %) et à une précision/un rappel très faibles. Ceci a été observé pendant son processus d'apprentissage et l'évaluation finale a confirmé ces mauvaises performances. Les modèles d'apprentissage par transfert ont largement surpassé le modèle CNN à partir de zéro.

### Matrice de confusion (apprentissage par transfert - optimiseur Adam sur l'ensemble de test)

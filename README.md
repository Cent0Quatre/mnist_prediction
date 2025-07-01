# Réseau de Neurones Multi-Couches pour la Classification MNIST

## Présentation du Projet

Ce projet implémente un réseau de neurones multi-couches (perceptron multicouche) pour la classification de chiffres manuscrits du dataset MNIST. L'implémentation est réalisée entièrement en Python/NumPy sans utilisation de frameworks de deep learning, démontrant une compréhension approfondie des mécanismes fondamentaux de l'apprentissage automatique.

## Architecture du Réseau

### Topologie
- **Couche d'entrée** : 784 neurones (images 28×28 pixels)
- **Couche cachée 1** : 32 neurones avec activation ReLU
- **Couche cachée 2** : 16 neurones avec activation ReLU  
- **Couche de sortie** : 10 neurones avec activation Softmax (classification 10 classes)

### Fonctions d'Activation
- **ReLU** (Rectified Linear Unit) : f(x) = max(0, x)
  - Utilisée dans les couches cachées pour introduire la non-linéarité
  - Avantage : évite le problème du gradient vanishing
- **Softmax** : f(x_i) = exp(x_i) / Σ(exp(x_j))
  - Utilisée en sortie pour obtenir une distribution de probabilités

## Implémentation Mathématique

### Propagation Avant (Forward Propagation)
```
Z₁ = W₁X + b₁
A₁ = ReLU(Z₁)
Z₂ = W₂A₁ + b₂
A₂ = ReLU(Z₂)
Z₃ = W₃A₂ + b₃
A₃ = Softmax(Z₃)
```

### Propagation Arrière (Backpropagation)
Calcul des gradients par application de la règle de dérivation en chaîne :
- **Couche de sortie** : dZ₃ = A₃ - Y_one_hot
- **Couches cachées** : dZ_l = W_{l+1}ᵀ · dZ_{l+1} ⊙ g'(Z_l)

### Optimisation
- **Algorithme** : Descente de gradient stochastique (SGD) avec mini-batches
- **Fonction de coût** : Cross-entropy loss
- **Initialisation** : Xavier/Glorot pour éviter l'explosion/disparition des gradients

## Fonctionnalités Techniques

### Préprocessing des Données
- Normalisation des pixels : [0, 255] → [0, 1]
- Mélange aléatoire des données d'entraînement
- Encodage one-hot des labels

### Optimisations Implémentées
- **Mini-batch SGD** : Compromis entre efficacité computationnelle et convergence
- **Initialisation He** : Adaptée aux fonctions ReLU
- **Stabilisation numérique** : Évitement des underflows/overflows

## Interface Web Interactive

### Technologies Utilisées
- **Backend** : Flask (Python)
- **Frontend** : HTML5 Canvas, JavaScript ES6
- **Visualisation** : SVG pour les connexions neuronales

### Fonctionnalités
- Dessin interactif de chiffres sur canvas 28×28
- Prédiction en temps réel
- Visualisation des activations neuronales par couche
- Affichage des connexions pondérées entre neurones

## Structure du Code

### Modules Principaux
- `learn.py` : Implémentation du réseau de neurones
- `main.py` : Serveur Flask et API REST
- `test.py` : Scripts de test et validation
- `static/` : Interface utilisateur (HTML/CSS/JS)

### Fonctions Clés
- `forward_prop()` : Propagation avant
- `backward_prop()` : Calcul des gradients
- `sgd_optimizer()` : Optimisation par descente de gradient
- `get_layer_activations()` : Extraction des activations pour visualisation

## Métriques de Performance

### Évaluation
- **Accuracy** : Pourcentage de classifications correctes
- **Confidence** : Probabilité associée à la prédiction
- **Validation** : Utilisation du dataset MNIST test

### Résultats Attendus
- Précision d'entraînement : ~95-98%
- Précision de validation : ~90-95%
- Temps de convergence : 20 époques

## Aspects Pédagogiques

### Concepts Démontrés
1. **Mathématiques** :
   - Algèbre linéaire (produits matriciels, transposées)
   - Calcul différentiel (dérivées partielles, règle de dérivation en chaîne)
   - Optimisation (descente de gradient)

2. **Informatique** :
   - Programmation orientée objet
   - Optimisation algorithmique
   - Développement web full-stack

3. **Machine Learning** :
   - Réseaux de neurones artificiels
   - Apprentissage supervisé
   - Régularisation et généralisation

### Défis Techniques Surmontés
- Implémentation from scratch sans frameworks
- Gestion de la stabilité numérique
- Visualisation interactive des états internes
- Optimisation des performances computationnelles

## Installation et Utilisation

### Prérequis
```bash
pip install numpy pandas matplotlib flask
```

### Exécution
```bash
python main.py
```
Interface accessible sur `http://localhost:5000`

### Tests
```bash
python test.py
```

import numpy as np
import pandas as pd
import sys, os
import matplotlib.pyplot as plt

# ---------------------------
# Chargement et préparation des données
# ---------------------------
data_train = pd.read_csv('./data/mnist_train.csv')
data_train = np.array(data_train)
np.random.shuffle(data_train)  # Mélange des données avant séparation

# Séparation des features et des labels
data_train = data_train.T
Y_train = data_train[0]  # Labels
X_train = data_train[1:] / 255.  # Normalisation des pixels
m_train = X_train.shape[1]

# Chargement des données de test (utilisées comme validation)
data_dev = pd.read_csv('./data/mnist_test.csv')
data_dev = np.array(data_dev).T
Y_dev = data_dev[0]  # Labels
X_dev = data_dev[1:] / 255.  # Normalisation des pixels

# ---------------------------
# Fonctions d'activation et softmax
# ---------------------------
def ReLU(Z: np.ndarray) -> np.ndarray:
    return np.maximum(Z, 0)

def dReLU(Z: np.ndarray) -> np.ndarray:
    return (Z > 0).astype(float)

def softmax(Z: np.ndarray) -> np.ndarray:
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

# ---------------------------
# Encodage One-Hot
# ---------------------------
def one_hot(Y: np.ndarray) -> np.ndarray:
    one_hot_Y = np.zeros((Y.size, 10))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

# ---------------------------
# Propagation avant
# ---------------------------
def forward_prop(W1: np.ndarray, b1: np.ndarray, 
                 W2: np.ndarray, b2: np.ndarray, 
                 W3: np.ndarray, b3: np.ndarray, 
                 X: np.ndarray) -> tuple[np.ndarray]:
    Z1 = np.dot(W1, X) + b1
    A1 = ReLU(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = ReLU(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

# ---------------------------
# Propagation arrière
# ---------------------------
def backward_prop(Z1: np.ndarray, A1: np.ndarray, 
                  Z2: np.ndarray, A2: np.ndarray, 
                  A3: np.ndarray, 
                  W2: np.ndarray, W3: np.ndarray,
                  X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray]:
    m = X.shape[1]
    one_hot_Y_mat = one_hot(Y)
    
    # Couche de sortie
    dZ3 = A3 - one_hot_Y_mat
    dW3 = (1/m) * np.dot(dZ3, A2.T)
    db3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)
    
    # Couche cachée 2
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = dA2 * dReLU(Z2)
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    
    # Couche cachée 1
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * dReLU(Z1)
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2, dW3, db3

# --------------------------------
# Initialisation des paramètres
# --------------------------------
def init_params() -> tuple[np.ndarray]:
    np.random.seed(42)
    # Réduction du nombre de neurones :
    #  - Couche 1 : 64 neurones
    #  - Couche 2 : 32 neurones
    #  - Couche de sortie : 10 neurones (pour MNIST)
    W1 = np.random.randn(32, 784) * np.sqrt(2 / 784)
    b1 = np.zeros((32, 1))
    W2 = np.random.randn(16, 32) * np.sqrt(2 / 32)
    b2 = np.zeros((16, 1))
    W3 = np.random.randn(10, 16) * np.sqrt(2 / 16)
    b3 = np.zeros((10, 1))
    return W1, b1, W2, b2, W3, b3

# ---------------------------
# Mise à jour des paramètres
# ---------------------------
def update_params(W1: np.ndarray, b1: np.ndarray, 
                  W2: np.ndarray, b2: np.ndarray, 
                  W3: np.ndarray, b3: np.ndarray, 
                  dW1: np.ndarray, db1: np.ndarray, 
                  dW2: np.ndarray, db2: np.ndarray, 
                  dW3: np.ndarray, db3: np.ndarray, 
                  alpha: float) -> tuple[np.ndarray]:
    W1 -= alpha * dW1
    b1 -= alpha * db1    
    W2 -= alpha * dW2  
    b2 -= alpha * db2
    W3 -= alpha * dW3  
    b3 -= alpha * db3    
    return W1, b1, W2, b2, W3, b3

# ---------------------------
# Fonctions pour les prédictions et l'évaluation
# ---------------------------
def get_predictions(A3: np.ndarray) -> np.ndarray:
    return np.argmax(A3, axis=0)

def make_predictions(X: np.ndarray, 
                     W1: np.ndarray, b1: np.ndarray, 
                     W2: np.ndarray, b2: np.ndarray, 
                     W3: np.ndarray, b3: np.ndarray) -> np.ndarray:
    A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)[5]
    predictions = get_predictions(A3)
    return predictions

def test_prediction_with_confidence(img: np.ndarray, 
                                  W1: np.ndarray, b1: np.ndarray, 
                                  W2: np.ndarray, b2: np.ndarray, 
                                  W3: np.ndarray, b3: np.ndarray) -> tuple[int, float]:
    # Propagation avant pour obtenir les activations finales
    A3 = forward_prop(W1, b1, W2, b2, W3, b3, img)[5]
    
    # Obtenir la prédiction (classe avec la plus haute probabilité)
    prediction = get_predictions(A3)[0]
    
    # Obtenir la probabilité associée à la prédiction
    confidence = float(np.max(A3))  # Prend la plus haute probabilité
    
    return prediction, confidence

def test_prediction(index: int, 
                    W1: np.ndarray, b1: np.ndarray, 
                    W2: np.ndarray, b2: np.ndarray, 
                    W3: np.ndarray, b3: np.ndarray) -> None:
    current_image = X_train[:, index, None]
    prediction, confidence = test_prediction_with_confidence(current_image, W1, b1, W2, b2, W3, b3)
    label = Y_train[index]
    print("Prediction:", prediction)
    print("Confidence: {:.2%}".format(confidence))  # Affiche la probabilité en pourcentage
    print("Label:", label)
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

def get_accuracy(predictions: np.ndarray, Y: np.ndarray) -> float:
    return np.sum(predictions == Y) / Y.size

# -----------------------
# Process server
# -----------------------
def get_layer_activations(img: np.ndarray, W1: np.ndarray, b1: np.ndarray, 
                         W2: np.ndarray, b2: np.ndarray, 
                         W3: np.ndarray, b3: np.ndarray) -> tuple:
    # Propagation avant pour obtenir toutes les activations
    Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, img)
    
    # Obtenir la prédiction et la confiance
    prediction = get_predictions(A3)[0]
    confidence = float(np.max(A3))
    
    # Normaliser les activations de chaque couche entre 0 et 1
    A1_norm = (A1 - A1.min()) / (A1.max() - A1.min() + 1e-10)
    A2_norm = (A2 - A2.min()) / (A2.max() - A2.min() + 1e-10)
    A3_norm = A3  # A3 est déjà normalisé (softmax)
    
    return {
        'prediction': int(prediction),
        'confidence': float(confidence),
        'layer1_activations': A1_norm.flatten().tolist(),  # 32 neurones
        'layer2_activations': A2_norm.flatten().tolist(),  # 16 neurones
        'layer3_activations': A3_norm.flatten().tolist()   # 10 neurones (sortie)
    }
# -----------------------
# Descente de Gradient
# -----------------------
def create_mini_batches(X: np.ndarray, Y: np.ndarray, batch_size: int) -> list:
    """
    Crée des mini-batches à partir des données d'entrée
    """
    m = X.shape[1]
    mini_batches = []
    
    # Mélanger les données
    permutation = np.random.permutation(m)
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[permutation]
    
    # Créer les mini-batches
    num_complete_batches = m // batch_size
    for k in range(num_complete_batches):
        mini_batch_X = shuffled_X[:, k * batch_size:(k + 1) * batch_size]
        mini_batch_Y = shuffled_Y[k * batch_size:(k + 1) * batch_size]
        mini_batches.append((mini_batch_X, mini_batch_Y))
    
    # Gérer le dernier batch s'il est incomplet
    if m % batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_batches * batch_size:]
        mini_batch_Y = shuffled_Y[num_complete_batches * batch_size:]
        mini_batches.append((mini_batch_X, mini_batch_Y))
    
    return mini_batches

def compute_cost(A3: np.ndarray, Y: np.ndarray) -> float:
    """
    Calcule la cross-entropy loss
    """
    m = Y.shape[0]
    one_hot_Y = one_hot(Y)
    log_probs = np.multiply(np.log(A3 + 1e-15), one_hot_Y)
    cost = -(1/m) * np.sum(log_probs)
    return float(cost)

def sgd_optimizer(X: np.ndarray, Y: np.ndarray, learning_rate: float, epochs: int, batch_size: int, print_cost: bool = True) -> None:
    """
    Implémentation de SGD avec mini-batches
    """
    W1, b1, W2, b2, W3, b3 = init_params()
    costs = []
    
    for epoch in range(epochs):
        epoch_cost = 0
        mini_batches = create_mini_batches(X, Y, batch_size)
        num_batches = len(mini_batches)
        
        for i, (mini_batch_X, mini_batch_Y) in enumerate(mini_batches):
            # Forward propagation
            Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, mini_batch_X)
            # Backward propagation
            dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, A3, W2, W3, mini_batch_X, mini_batch_Y)
            # Mise à jour des paramètres
            W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learning_rate)
            
            # Calculer le coût pour ce mini-batch
            batch_cost = compute_cost(A3, mini_batch_Y)
            epoch_cost += batch_cost / num_batches
            # Afficher la progression
            if print_cost:
                progress = (epoch * num_batches + i + 1) / (epochs * num_batches) * 100
                print(f"Progress: {progress:.2f}%  Current cost: {batch_cost:.4f}")
                sys.stdout.write("\033[F")
                sys.stdout.write("\033[K")
        
        costs.append(epoch_cost)
        if print_cost:
            print(f"Epoch {epoch + 1}/{epochs} - Cost: {epoch_cost:.4f}")
    
    # Sauvegarder les paramètres
    np.save("./weights/W1.npy", W1)
    np.save("./weights/b1.npy", b1)
    np.save("./weights/W2.npy", W2)
    np.save("./weights/b2.npy", b2)
    np.save("./weights/W3.npy", W3)
    np.save("./weights/b3.npy", b3)

# -----------------------
# Initialisation
# -----------------------
def load_model():
    if not all(os.path.exists(f) for f in ["weights/W1.npy", "weights/b1.npy", "weights/W2.npy", "weights/b2.npy", "weights/W3.npy", "weights/b3.npy"]):
        sgd_optimizer(X=X_train, Y=Y_train, learning_rate=0.1, epochs=20, batch_size=64, print_cost=True)
    
    W1 = np.load("./weights/W1.npy")
    b1 = np.load("./weights/b1.npy")
    W2 = np.load("./weights/W2.npy")
    b2 = np.load("./weights/b2.npy")
    W3 = np.load("./weights/W3.npy")
    b3 = np.load("./weights/b3.npy")
    return W1, b1, W2, b2, W3, b3

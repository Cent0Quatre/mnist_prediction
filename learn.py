import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------
# Chargement et préparation des données
# ---------------------------
data = pd.read_csv('./data/train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)  # Mélange des données avant séparation

# Ensemble de développement (validation)
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n] / 255.

# Ensemble d'entraînement
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n] / 255.
m_train = X_train.shape[1]

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
# Propagation avant (forward propagation)
# ---------------------------
def forward_prop(W1: np.ndarray, b1: np.ndarray, 
                 W2: np.ndarray, b2: np.ndarray, 
                 W3: np.ndarray, b3: np.ndarray, 
                 X: np.ndarray) -> tuple[np.ndarray, ...]:
    Z1 = np.dot(W1, X) + b1
    A1 = ReLU(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = ReLU(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

# ---------------------------
# Propagation arrière (backward propagation)
# ---------------------------
def backward_prop(Z1: np.ndarray, A1: np.ndarray, 
                  Z2: np.ndarray, A2: np.ndarray, 
                  A3: np.ndarray, 
                  W2: np.ndarray, W3: np.ndarray,
                  X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, ...]:
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

# ---------------------------
# Initialisation des paramètres (avec He initialization)
# ---------------------------
def init_params() -> tuple[np.ndarray, ...]:
    np.random.seed(42)
    # Réduction du nombre de neurones :
    #  - Couche 1 : 32 neurones
    #  - Couche 2 : 16 neurones
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
                  alpha: float) -> tuple[np.ndarray, ...]:
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2
    W3 = W3 - alpha * dW3  
    b3 = b3 - alpha * db3    
    return W1, b1, W2, b2, W3, b3

# ---------------------------
# Descente de gradient
# ---------------------------
def gradient_descent(X: np.ndarray, Y: np.ndarray, alpha: float, iterations: int) -> None:
    W1, b1, W2, b2, W3, b3 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, A3, W2, W3, X, Y)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3,
                                                 dW1, db1, dW2, db2, dW3, db3, alpha)
        # Affichage du progrès
        print(round((i / iterations) * 100, 2), "%")
        sys.stdout.write("\033[F")  # Remonte d'une ligne
        sys.stdout.write("\033[K")  # Efface la ligne
    # Sauvegarde des paramètres
    np.save("./weights/W1.npy", W1)
    np.save("./weights/b1.npy", b1)
    np.save("./weights/W2.npy", W2)
    np.save("./weights/b2.npy", b2)
    np.save("./weights/W3.npy", W3)
    np.save("./weights/b3.npy", b3)

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
    """
    Fait une prédiction sur un exemple et renvoie la prédiction ainsi que sa probabilité associée.
    
    Paramètres:
        index (int): L'index de l'exemple à tester
        W1, b1, W2, b2, W3, b3 (np.ndarray): Les paramètres du modèle
        
    Retourne:
        tuple[int, float]: (prédiction, probabilité)
    """
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
    print(current_image)
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

# -------------------
# Process images
# -------------------

def load_and_preprocess_image(image_path: str) -> np.ndarray:
    # Charger l'image
    img = Image.open(image_path)
    
    # Vérifier la taille
    if img.size != (28, 28):
        raise ValueError(f"L'image doit être de taille 28x28 pixels. Taille actuelle: {img.size}")
    
    # Convertir en niveaux de gris
    img = img.convert('L')
    
    # Convertir en tableau numpy
    img_array = np.array(img, dtype=np.float32)
    
    # Inverser les valeurs (pour que 0 = blanc et 1 = noir)
    img_array = img_array / 255.0
    return img_array.reshape(784, 1)    

def test_img_prediction(img: str, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray, W3: np.ndarray, b3: np.ndarray) -> None:
    img = load_and_preprocess_image(img)
    prediction, confidence = test_prediction_with_confidence(img, W1, b1, W2, b2, W3, b3)
    print("Prediction:", prediction)
    print("Confidence: {:.2%}".format(confidence))  # Affiche la probabilité en pourcentage

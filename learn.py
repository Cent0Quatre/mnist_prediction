import numpy as np
import pandas as pd
import sys

data = pd.read_csv('./data/train.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255
m_train = X_train.shape[1]

###############################################################
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z: np.ndarray) -> np.ndarray:
    return np.maximum(Z, 0)

def softmax(Z: np.ndarray) -> np.ndarray:
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray, X: np.ndarray) -> tuple[np.ndarray]:
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def dReLU(Z: np.ndarray) -> np.ndarray:
    return Z > 0

def one_hot(Y: np.ndarray) -> np.ndarray:
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1: np.ndarray, A1: np.ndarray, A2: np.ndarray, W2: np.ndarray, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray]:
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * dReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray, dW1: np.ndarray, db1: np.ndarray, dW2: np.ndarray, db2: np.ndarray, alpha: float) -> tuple[np.ndarray]:
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

##############################################
def get_predictions(A2: np.ndarray) -> np.ndarray:
    return np.argmax(A2, 0)

def get_accuracy(predictions: np.ndarray, Y: np.ndarray) -> float:
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X: np.ndarray, Y: np.ndarray, alpha: float, iterations: int) -> None:
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, _, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        print(round((i/iterations)*100, 2), "%")
        # Supprime la ligne précédente
        sys.stdout.write("\033[F")  # Déplace le curseur une ligne au-dessus
        sys.stdout.write("\033[K")  # Efface la ligne entière

    np.save("./weights/W1.npy", W1)
    np.save("./weights/b1.npy", b1)
    np.save("./weights/W2.npy", W2)
    np.save("./weights/b2.npy", b2)

###############################################
def make_predictions(X: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    A2 = forward_prop(W1, b1, W2, b2, X)[3]
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index: int, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> None:
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

from learn import *
import os

idx = np.random.randint(0, 10000)

if not all(os.path.exists(f) for f in ["weights/W1.npy", "weights/b1.npy", "weights/W2.npy", "weights/b2.npy"]):
    gradient_descent(X_train, Y_train, 0.1, 500)

W1 = np.load("./weights/W1.npy")
b1 = np.load("./weights/b1.npy")
W2 = np.load("./weights/W2.npy")
b2 = np.load("./weights/b2.npy")

test_prediction(idx, W1, b1, W2, b2)

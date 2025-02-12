from learn import *
import os

if not all(os.path.exists(f) for f in ["weights/W1.npy", "weights/b1.npy", "weights/W2.npy", "weights/b2.npy", "weights/W3.npy", "weights/b3.npy"]):
    gradient_descent(X_train, Y_train, 0.1, 500)

W1 = np.load("./weights/W1.npy")
b1 = np.load("./weights/b1.npy")
W2 = np.load("./weights/W2.npy")
b2 = np.load("./weights/b2.npy")
W3 = np.load("./weights/W3.npy")
b3 = np.load("./weights/b3.npy")

# print(get_accuracy(make_predictions(X_train, W1, b1, W2, b2, W3, b3), Y_train))

print(load_and_preprocess_image("./dessin-28x28.png"))
test_img_prediction("./dessin-28x28.png", W1, b1, W2, b2, W3, b3)

#test_prediction(343, W1, b1, W2, b2, W3, b3)
# LE PROBLEME VIENT DE LA CONVERSION DE L4IMAGE NUL A CIER

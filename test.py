from learn import *

W1, b1, W2, b2, W3, b3 = load_model()
print(get_accuracy(make_predictions(X_train, W1, b1, W2, b2, W3, b3), Y_train))

idx = np.random.randint(0, 10000)
test_prediction(idx, W1, b1, W2, b2, W3, b3)

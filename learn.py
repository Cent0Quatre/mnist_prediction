import numpy as np
import csv
import tkinter as tk
from tkinter import messagebox

# 📥 Fonction pour charger MNIST CSV
def load_mnist_csv(filename):
    data = []
    labels = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Ignorer l'en-tête
        for row in reader:
            labels.append(int(row[0]))  # Première colonne = label (0-9)
            pixels = np.array(row[1:], dtype=np.float32) / 255.0  # Normalisation 0-255 → 0-1
            data.append(pixels)
    return np.array(data), np.array(labels)

# 📥 Charger toutes les données
x_train, y_train = load_mnist_csv("mnist_train.csv")
x_test, y_test = load_mnist_csv("mnist_test.csv")

# 🚀 Perceptron avec numpy
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=10, n_classes=10):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.n_classes = n_classes
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((self.n_classes, n_features))  # Un poids pour chaque classe
        self.bias = np.zeros(self.n_classes)  # Un biais pour chaque classe

        for _ in range(self.n_iters):
            for idx in range(n_samples):
                linear_output = np.dot(self.weights, X[idx]) + self.bias  # Produit scalaire pour chaque classe
                y_predicted = np.argmax(linear_output)  # Classe prédite (avec la plus grande sortie)
                
                if y_predicted != y[idx]:  # Mise à jour si l'étiquette est incorrecte
                    # Mise à jour des poids et du biais pour la classe correcte et incorrecte
                    self.weights[y[idx]] += self.lr * X[idx]
                    self.bias[y[idx]] += self.lr
                    self.weights[y_predicted] -= self.lr * X[idx]
                    self.bias[y_predicted] -= self.lr

    def predict(self, X):
        linear_output = np.dot(self.weights, X.T) + self.bias[:, np.newaxis]  # Calcul pour chaque exemple
        return np.argmax(linear_output, axis=0)  # Prédiction avec la plus grande sortie pour chaque exemple

# 📊 Entraînement du Perceptron
perceptron = Perceptron(learning_rate=0.1, n_iters=10)
perceptron.fit(x_train, y_train)

# 📈 Évaluation
y_pred = perceptron.predict(x_test)
accuracy = np.mean(y_pred == y_test)
print(f"Précision du Perceptron sur MNIST (0-9) : {accuracy:.2%}")

# 🎨 Interface Graphique pour dessiner avec la souris (Tkinter)
class DrawApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dessinez un chiffre (0-9)")
        
        self.canvas = tk.Canvas(self.root, width=280, height=280, bg="white")
        self.canvas.pack()
        
        self.canvas.bind("<B1-Motion>", self.paint)  # Dessiner en maintenant le bouton gauche de la souris
        self.canvas.bind("<ButtonRelease-1>", self.save_drawing)  # Sauvegarder à la fin du dessin

        self.last_x, self.last_y = None, None
        self.drawing = np.zeros((28, 28))  # 28x28 image de dessin
        self.is_eraser = False  # Mode gomme désactivé par défaut

        self.button_predict = tk.Button(self.root, text="Prédire", command=self.predict)
        self.button_predict.pack()

        self.button_eraser = tk.Button(self.root, text="Gomme", command=self.toggle_eraser)
        self.button_eraser.pack()

    def paint(self, event):
        """ Dessine un point ou efface selon le mode. """
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            if self.is_eraser:
                self.canvas.create_line(self.last_x, self.last_y, x, y, width=10, fill="white", capstyle=tk.ROUND, smooth=True)
                self.update_drawing(x, y, erase=True)
            else:
                self.canvas.create_line(self.last_x, self.last_y, x, y, width=10, fill="black", capstyle=tk.ROUND, smooth=True)
                self.update_drawing(x, y, erase=False)
        self.last_x, self.last_y = x, y

    def save_drawing(self, event):
        """ Sauvegarde l'image après avoir fini de dessiner. """
        self.last_x, self.last_y = None, None

    def update_drawing(self, x, y, erase=False):
        """ Met à jour le tableau numpy représentant le dessin (efface ou dessine). """
        row = int(y // 10)  # Divise par 10 pour obtenir l'index des pixels
        col = int(x // 10)  # Divise par 10 pour obtenir l'index des pixels
        if 0 <= row < 28 and 0 <= col < 28:
            if erase:
                self.drawing[row, col] = 0.0  # Efface le pixel
            else:
                self.drawing[row, col] = 1.0  # Dessine un pixel noir

    def toggle_eraser(self):
        """ Bascule entre le mode dessin et le mode gomme. """
        self.is_eraser = not self.is_eraser
        if self.is_eraser:
            self.button_eraser.config(bg="red")  # Indique que la gomme est activée
        else:
            self.button_eraser.config(bg="SystemButtonFace")  # Rétablit le bouton normal

    def predict(self):
        """ Prédit le chiffre dessiné. """
        # Aplatir l'image pour la passer au perceptron
        flat_drawing = self.drawing.flatten()

        # Prédiction
        prediction = perceptron.predict(np.array([flat_drawing]))
        messagebox.showinfo("Prédiction", f"Le chiffre que vous avez dessiné est : {prediction[0]}")

# Lancer l'application Tkinter
root = tk.Tk()
app = DrawApp(root)
root.mainloop()

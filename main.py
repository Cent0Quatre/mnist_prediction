from flask import Flask, render_template, request, jsonify
from learn import *

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    pixel_data = request.json.get('pixels', [])
    img_array = np.array(pixel_data, dtype=np.float32) / 255.0
    img_array = img_array.reshape(784, 1)
    
    W1, b1, W2, b2, W3, b3 = load_model()
    
    # Obtenir toutes les activations
    activations = get_layer_activations(img_array, W1, b1, W2, b2, W3, b3)
    
    return jsonify(activations)

if __name__ == "__main__":
    app.run(debug=True)

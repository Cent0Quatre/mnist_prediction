import { createNetworkSVG, updateConnections } from './connections.js';

// Éléments du DOM et constantes
const canvas = document.getElementById("pixelCanvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const gridSize = 28;
const pixelSize = canvas.width / gridSize;
let isDrawing = false;
let lastX = -1, lastY = -1;
let drawnPixels = new Map();
let isErasing = false;

// Création de l'interface des jauges
function createGauges() {
    // Couche 1 : 32 neurones
    const layer1 = document.getElementById('layer1');
    for (let i = 0; i < 32; i++) {
        const jauge = createJauge();
        layer1.appendChild(jauge);
    }
    
    // Couche 2 : 16 neurones
    const layer2 = document.getElementById('layer2');
    for (let i = 0; i < 16; i++) {
        const jauge = createJauge();
        layer2.appendChild(jauge);
    }
    
    // Couche de sortie : 10 neurones
    const layer3 = document.getElementById('layer3');
    for (let i = 0; i < 10; i++) {
        const jauge = createJauge();
        const label = document.createElement('div');
        label.style.textAlign = 'center';
        label.style.fontSize = '12px';
        label.textContent = i;
        const container = document.createElement('div');
        container.style.display = 'flex';
        container.style.flexDirection = 'column';
        container.style.alignItems = 'center';
        container.style.gap = '2px';
        container.appendChild(jauge);
        container.appendChild(label);
        layer3.appendChild(container);
    }
}

// Création d'une jauge individuelle
function createJauge() {
    const jauge = document.createElement('div');
    jauge.className = 'jauge';
    const remplissage = document.createElement('div');
    remplissage.className = 'remplissage';
    jauge.appendChild(remplissage);
    return jauge;
}

// Mise à jour des activations pour une couche
function updateActivations(layerId, activations) {
    const jauges = document.querySelectorAll(`#${layerId} .jauge .remplissage`);
    activations.forEach((value, index) => {
        if (jauges[index]) {
            jauges[index].style.height = `${value * 100}%`;
        }
    });
    
    // Mettre à jour les connexions si toutes les activations sont disponibles
    if (window.currentActivations === undefined) {
        window.currentActivations = {};
    }
    window.currentActivations[layerId] = activations;
    
    if (window.currentActivations.layer1 && 
        window.currentActivations.layer2 && 
        window.currentActivations.layer3) {
        updateConnections(
            window.currentActivations.layer1,
            window.currentActivations.layer2,
            window.currentActivations.layer3
        );
    }
}
// Mise à jour des prédictions
function updatePrediction() {
    let grayValues = new Array(gridSize * gridSize).fill(0);
    
    drawnPixels.forEach((intensity, key) => {
        const [x, y] = key.split(',').map(Number);
        const index = y * gridSize + x;
        if (index >= 0 && index < gridSize * gridSize) {
            grayValues[index] = Math.round(intensity * 255);
        }
    });

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ pixels: grayValues })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('prediction').textContent = 
            `Prédiction: ${data.prediction} (${(data.confidence * 100).toFixed(1)}%)`;
        updateActivations('layer1', data.layer1_activations);
        updateActivations('layer2', data.layer2_activations);
        updateActivations('layer3', data.layer3_activations);
    })
    .catch(error => console.error('Error:', error));
}

// Dessin de la grille
function drawGrid() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#ccc";
    for (let i = 0; i <= gridSize; i++) {
        let pos = i * pixelSize;
        ctx.beginPath();
        ctx.moveTo(pos, 0);
        ctx.lineTo(pos, canvas.height);
        ctx.moveTo(0, pos);
        ctx.lineTo(canvas.width, pos);
        ctx.stroke();
    }
    lastX = -1;
    lastY = -1;
    drawnPixels.clear();
    document.getElementById('prediction').textContent = 'Prédiction: -';
    
    // Réinitialisation des jauges
    document.querySelectorAll('.remplissage').forEach(jauge => {
        jauge.style.height = '0%';
    });
}

// Effacement d'un pixel et de ses voisins
function erasePixel(x, y) {
    if (x < 0 || x >= gridSize || y < 0 || y >= gridSize) {
        return;
    }

    if (x === lastX && y === lastY) return;
    
    lastX = x;
    lastY = y;

    const pixels = [
        [x, y],
        [x, y - 1], [x - 1, y], [x + 1, y], [x, y + 1],
        [x - 1, y - 1], [x + 1, y - 1], [x - 1, y + 1], [x + 1, y + 1]
    ];

    pixels.forEach(([px, py]) => {
        if (px >= 0 && px < gridSize && py >= 0 && py < gridSize) {
            drawnPixels.delete(`${px},${py}`);
            ctx.clearRect(px * pixelSize, py * pixelSize, pixelSize, pixelSize);
            
            ctx.strokeStyle = "#ccc";
            ctx.beginPath();
            ctx.moveTo(px * pixelSize, py * pixelSize);
            ctx.lineTo((px + 1) * pixelSize, py * pixelSize);
            ctx.moveTo(px * pixelSize, py * pixelSize);
            ctx.lineTo(px * pixelSize, (py + 1) * pixelSize);
            ctx.stroke();
        }
    });

    updatePrediction();
}

// Remplissage d'un pixel et de ses voisins
function fillPixel(x, y) {
    if (x < 0 || x >= gridSize || y < 0 || y >= gridSize) {
        return;
    }

    if (x === lastX && y === lastY) return;
    
    lastX = x;
    lastY = y;

    drawnPixels.set(`${x},${y}`, 0.85);
    ctx.fillStyle = "rgba(0, 0, 0, 0.85)";
    ctx.fillRect(x * pixelSize, y * pixelSize, pixelSize, pixelSize);

    const directNeighbors = [
        [x, y - 1], [x - 1, y], [x + 1, y], [x, y + 1]
    ];

    const diagonalNeighbors = [
        [x - 1, y - 1], [x + 1, y - 1], 
        [x - 1, y + 1], [x + 1, y + 1]
    ];

    ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
    directNeighbors.forEach(([nx, ny]) => {
        if (nx >= 0 && nx < gridSize && ny >= 0 && ny < gridSize) {
            drawnPixels.set(`${nx},${ny}`, 0.5);
            ctx.fillRect(nx * pixelSize, ny * pixelSize, pixelSize, pixelSize);
        }
    });

    ctx.fillStyle = "rgba(0, 0, 0, 0.25)";
    diagonalNeighbors.forEach(([nx, ny]) => {
        if (nx >= 0 && nx < gridSize && ny >= 0 && ny < gridSize) {
            drawnPixels.set(`${nx},${ny}`, 0.25);
            ctx.fillRect(nx * pixelSize, ny * pixelSize, pixelSize, pixelSize);
        }
    });

    updatePrediction();
}

// Gestion du mouvement de la souris
function handleMouseMove(e) {
    if (!isDrawing) return;

    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / pixelSize);
    const y = Math.floor((e.clientY - rect.top) / pixelSize);

    if (x >= 0 && x < gridSize && y >= 0 && y < gridSize) {
        if (isErasing) {
            erasePixel(x, y);
        } else {
            fillPixel(x, y);
        }
    }
}

// Gestion du clic de la souris
function handleMouseDown(e) {
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / pixelSize);
    const y = Math.floor((e.clientY - rect.top) / pixelSize);

    if (x >= 0 && x < gridSize && y >= 0 && y < gridSize) {
        isDrawing = true;
        handleMouseMove(e);
    }
}

// Mise à jour du style du bouton de gomme
function updateEraseButtonStyle() {
    const eraseButton = document.getElementById('eraseButton');
    eraseButton.style.backgroundColor = isErasing ? '#ff9999' : '';
    canvas.style.cursor = isErasing ? 'crosshair' : 'default';
}

// Event Listeners
canvas.addEventListener("mousedown", handleMouseDown);
document.addEventListener("mouseup", () => {
    isDrawing = false;
});
canvas.addEventListener("mousemove", handleMouseMove);
canvas.addEventListener("mouseleave", () => {
    isDrawing = false;
});
document.getElementById("clearCanvas").addEventListener("click", drawGrid);
document.getElementById("eraseButton").addEventListener("click", () => {
    isErasing = !isErasing;
    updateEraseButtonStyle();
});

// Initialisation
createNetworkSVG();
createGauges();
drawGrid();
updateEraseButtonStyle();

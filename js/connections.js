// Constantes pour la visualisation
const LAYER_SIZES = [32, 16, 10];

// Création du SVG pour les connexions
function createNetworkSVG() {
    const networkContainer = document.querySelector('.network-viz');
    const svgContainer = document.createElement('div');
    svgContainer.className = 'connections-container';
    
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.id = 'connections-svg';
    svgContainer.appendChild(svg);
    
    // Insérer le SVG avant le premier layer
    networkContainer.insertBefore(svgContainer, networkContainer.firstChild);
    
    // Attendre que les jauges soient créées avant de dessiner les connexions
    setTimeout(() => {
        updateSVGSize();
        drawConnections();
    }, 100);
}

// Mise à jour de la taille du SVG
function updateSVGSize() {
    const svg = document.getElementById('connections-svg');
    const networkContainer = document.querySelector('.network-viz');
    
    svg.style.width = networkContainer.offsetWidth + 'px';
    svg.style.height = networkContainer.offsetHeight + 'px';
    svg.setAttribute('viewBox', `0 0 ${networkContainer.offsetWidth} ${networkContainer.offsetHeight}`);
}

// Obtenir les coordonnées réelles d'un neurone
function getNeuronCoordinates(layerIndex, neuronIndex) {
    const layerId = `layer${layerIndex + 1}`;
    const layer = document.getElementById(layerId);
    const jauges = layer.querySelectorAll('.jauge');
    const jauge = jauges[neuronIndex];
    
    if (!jauge) return null;

    const rect = jauge.getBoundingClientRect();
    const containerRect = document.querySelector('.network-viz').getBoundingClientRect();
    
    // Calculer les positions relatives au conteneur SVG
    const x = rect.left - containerRect.left + rect.width / 2;
    const y = rect.top - containerRect.top + rect.height / 2;
    
    return { x, y };
}

// Dessiner les connexions entre les neurones
function drawConnections() {
    const svg = document.getElementById('connections-svg');
    svg.innerHTML = ''; // Clear existing connections
    
    // Dessiner les connexions entre les couches
    for (let layer = 0; layer < LAYER_SIZES.length - 1; layer++) {
        const currentLayerSize = LAYER_SIZES[layer];
        const nextLayerSize = LAYER_SIZES[layer + 1];
        
        for (let i = 0; i < currentLayerSize; i++) {
            const start = getNeuronCoordinates(layer, i);
            if (!start) continue;
            
            for (let j = 0; j < nextLayerSize; j++) {
                const end = getNeuronCoordinates(layer + 1, j);
                if (!end) continue;
                
                const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                line.setAttribute('x1', start.x);
                line.setAttribute('y1', start.y);
                line.setAttribute('x2', end.x);
                line.setAttribute('y2', end.y);
                line.setAttribute('stroke', '#ddd');
                line.setAttribute('stroke-width', '0.5');
                line.setAttribute('data-start-layer', layer);
                line.setAttribute('data-start-neuron', i);
                line.setAttribute('data-end-layer', layer + 1);
                line.setAttribute('data-end-neuron', j);
                
                svg.appendChild(line);
            }
        }
    }
}

// Mettre à jour l'opacité et la couleur des connexions en fonction des activations
function updateConnections(layer1Act, layer2Act, layer3Act) {
    const allActivations = [layer1Act, layer2Act, layer3Act];
    const lines = document.querySelectorAll('#connections-svg line');
    
    lines.forEach(line => {
        const startLayer = parseInt(line.getAttribute('data-start-layer'));
        const startNeuron = parseInt(line.getAttribute('data-start-neuron'));
        const endLayer = parseInt(line.getAttribute('data-end-layer'));
        const endNeuron = parseInt(line.getAttribute('data-end-neuron'));
        
        const startActivation = allActivations[startLayer][startNeuron];
        const endActivation = allActivations[endLayer][endNeuron];
        
        // L'opacité de la connexion est basée sur la moyenne des activations
        const opacity = (startActivation + endActivation) / 2;
        line.style.opacity = opacity;
        
        // Changer la couleur en fonction de l'intensité
        const intensity = Math.floor(opacity * 255);
        const color = `rgb(0, ${intensity}, 0)`;
        line.setAttribute('stroke', color);
        line.setAttribute('stroke-width', opacity * 2); // Épaisseur variable
    });
}

// Event listener pour le redimensionnement
let resizeTimeout;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
        updateSVGSize();
        drawConnections();
    }, 250);
});

// Exporter les fonctions nécessaires
export { createNetworkSVG, updateConnections };

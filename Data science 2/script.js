// --- Configuración Three.js ---
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.getElementById('render3d').appendChild(renderer.domElement);

scene.background = new THREE.Color(0x000011);
scene.fog = new THREE.FogExp2(0x000011, 0.02);

const ambientLight = new THREE.AmbientLight(0x404040, 1.5);
scene.add(ambientLight);
const pointLight = new THREE.PointLight(0xffffff, 2, 50);
pointLight.position.set(5,5,5);
scene.add(pointLight);

const materialNormal = new THREE.MeshPhongMaterial({ color: 0x00ffcc, emissive: 0x004466, shininess: 100 });
const materialActiva = new THREE.MeshPhongMaterial({ color: 0xff2200, emissive: 0xff4400, shininess: 200 });

const inputGroup = new THREE.Group();
const nodosGroup = new THREE.Group();
const conexionesGroup = new THREE.Group();
scene.add(inputGroup, nodosGroup, conexionesGroup);

const inputSize = 10;
const modelo = { nodos_por_capa: [16, 32, 512, 128] };

const nodos = [];
const capas = [];

modelo.nodos_por_capa.forEach((cantidad, c) => {
    const capa = [];
    for (let i = 0; i < cantidad; i++) {
        const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.1, 16, 16), materialNormal.clone());
        const angle = (i / cantidad) * Math.PI * 2;
        const radius = 2 + c * 0.5;
        const x = Math.cos(angle) * radius + (Math.random() - 0.5) * 0.2;
        const y = Math.sin(angle) * radius + (Math.random() - 0.5) * 0.2;
        const z = c * 2;
        sphere.position.set(x, y, z);
        nodosGroup.add(sphere);
        nodos.push(sphere);
        capa.push(sphere);
    }
    capas.push(capa);
});

// --- Conexiones entre capas ---
const materialLinea = new THREE.LineBasicMaterial({ color: 0x006699, transparent: true, opacity: 0.3 });
for (let c = 0; c < capas.length - 1; c++) {
    capas[c].forEach(nodoO => {
        for (let i = 0; i < 5; i++) {
            const nodoD = capas[c+1][Math.floor(Math.random() * capas[c+1].length)];
            const geometry = new THREE.BufferGeometry().setFromPoints([nodoO.position, nodoD.position]);
            const line = new THREE.Line(geometry, materialLinea);
            conexionesGroup.add(line);
        }
    });
}

// --- Visualización del Input ---
let inputPixeles = [];
function crearInputFromImage(img) {
    inputPixeles.forEach(p => inputGroup.remove(p));
    inputPixeles = [];

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = inputSize;
    canvas.height = inputSize;
    ctx.drawImage(img, 0, 0, inputSize, inputSize);

    for (let y = 0; y < inputSize; y++) {
        for (let x = 0; x < inputSize; x++) {
            const data = ctx.getImageData(x, y, 1, 1).data;
            const brightness = (data[0] + data[1] + data[2]) / (3 * 255);
            const color = new THREE.Color(brightness, brightness, brightness);
            const sphere = new THREE.Mesh(
                new THREE.SphereGeometry(0.05),
                new THREE.MeshBasicMaterial({ color })
            );
            sphere.position.set((x - inputSize/2) * 0.15, (y - inputSize/2) * 0.15, 0);
            inputGroup.add(sphere);
            inputPixeles.push(sphere);
        }
    }
}

// --- Animaciones ---
function animarInputHaciaPrimeraCapa() {
    inputPixeles.forEach((pix, i) => {
        const destino = capas[0][i % capas[0].length];
        new TWEEN.Tween(pix.position).to(destino.position, 2000).delay(i * 20).start();
        new TWEEN.Tween(pix.scale).to({ x: 0.01, y: 0.01, z: 0.01 }, 2000).delay(i * 20).start();
    });
}

function activarNeurona(index) {
    const nodo = nodos[index];
    if (!nodo) return;
    nodo.material.color.set(0xff2200);
    nodo.material.emissive.set(0xff4400);
    new TWEEN.Tween(nodo.scale).to({ x: 1.5, y: 1.5, z: 1.5 }, 400).yoyo(true).repeat(1).onComplete(() => {
        nodo.material.color.set(0x00ffcc);
        nodo.material.emissive.set(0x004466);
    }).start();
}

function animarActivacionEntreCapas(origenIdx, destinoIdx, delay = 0) {
    const origen = capas[origenIdx];
    const destino = capas[destinoIdx];
    const intervalo = origenIdx === 2 ? 5 : 100;
    origen.forEach((nodo, i) => setTimeout(() => activarNeurona(nodos.indexOf(nodo)), delay + i * intervalo));
}

function animarActivacionFinal(delay = 0) {
    capas.at(-1).forEach((nodo, i) => {
        setTimeout(() => activarNeurona(nodos.indexOf(nodo)), delay + i * 50);
    });
}

function animarRedCompleta() {
    let delay = inputPixeles.length + 2200;
    animarInputHaciaPrimeraCapa();

    for (let c = 0; c < capas.length - 1; c++) {
        setTimeout(() => animarActivacionEntreCapas(c, c + 1), delay);
        delay += (c === 2 ? 500 : capas[c].length * 100 + 500);
    }
    setTimeout(() => animarActivacionFinal(), delay);
}

// --- Visualización Firma 3D ---
let firmaVisualGroup = new THREE.Group();
scene.add(firmaVisualGroup);

function mostrarFirmaEn3D(firmaB64) {
    firmaVisualGroup.clear();
    const firmaBin = atob(firmaB64);
    const offsetX = 8;

    for (let i = 0; i < firmaBin.length; i++) {
        const val = firmaBin.charCodeAt(i) / 255;
        const esfera = new THREE.Mesh(
            new THREE.SphereGeometry(0.05),
            new THREE.MeshBasicMaterial({ color: new THREE.Color(val, 0, 1 - val) })
        );
        esfera.position.set(offsetX, i * 0.12 - 3, 0);
        firmaVisualGroup.add(esfera);
    }
}

// --- Captura Genérica y Comunicación Backend ---
function capturarYEnviar(endpoint, extraData = {}, callback) {
    const video = document.getElementById('videoFeed');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    const imgData = canvas.toDataURL('image/jpeg');
    const payload = Object.assign({ imagen: imgData }, extraData);

    fetch(`http://localhost:5000/${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(r => r.json())
    .then(res => callback?.(res));
}

// --- Eventos de la Web ---
document.getElementById('imgInput').addEventListener('change', e => {
    const img = new Image();
    img.onload = () => {
        crearInputFromImage(img);
        animarRedCompleta();
    };
    img.src = URL.createObjectURL(e.target.files[0]);
});

function guardarFirmaDesdeWeb() {
    const nombre = document.getElementById('nombreInput').value.trim();
    if (!nombre) return alert("Ingrese un nombre válido");

    const video = document.getElementById('videoFeed');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    const imgData = canvas.toDataURL('image/jpeg');

    fetch('http://localhost:5000/guardar_firma', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ imagen: imgData, nombre })
    })
    .then(r => r.json())
    .then(data => {
        if (data.status === "ok") {
            alert("✅ Firma guardada correctamente");
            if (data.firma) mostrarFirmaEn3D(data.firma); 
            animarRedCompleta();
        } else {
            alert("⚠️ Error al guardar firma: " + (data.error || "desconocido"));
        }
    });
}


// --- Loop Principal ---
let frame = 0;
function animate(time) {
    requestAnimationFrame(animate);
    TWEEN.update(time);

    const ang = frame * 0.005;
    camera.position.set(Math.sin(ang) * 12, 8, Math.cos(ang) * 12);
    camera.lookAt(scene.position);

    nodos.forEach((n, i) => {
        const s = 1 + 0.02 * Math.sin(frame * 0.05 + i);
        n.scale.set(s, s, s);
    });

    renderer.render(scene, camera);
    frame++;
}

animate();
document.getElementById('videoFeed').src = 'http://localhost:5000/video_feed';

// --- Comparación periódica ---
setInterval(() => {
    capturarYEnviar('comparar', {}, (res) => {
        if (res.firma) {
            mostrarFirmaEn3D(res.firma);
            animarRedCompleta();
            console.log(res.conocido ? "Conocido" : "Desconocido", res.similaridad.toFixed(2) + "%");
        }
    });
}, 5000);

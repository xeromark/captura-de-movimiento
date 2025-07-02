// Elementos
const slide1 = document.getElementById('render3d'); // tu 3D canvas container
const slide2 = document.getElementById('slide2');
const btnSlide1 = document.getElementById('btnSlide1');
const btnSlide2 = document.getElementById('btnSlide2');

btnSlide1.onclick = () => {
  slide1.style.display = 'block';
  slide2.style.display = 'none';
};

btnSlide2.onclick = () => {
  slide1.style.display = 'none';
  slide2.style.display = 'block';
};

// Cargar imagen y procesar
const imgInput2 = document.getElementById('imgInput2');
const canvasOriginal = document.getElementById('canvasOriginal');
const canvasProcesado = document.getElementById('canvasProcesado');
const ctxOriginal = canvasOriginal.getContext('2d');
const ctxProcesado = canvasProcesado.getContext('2d');
const infoProcesamiento = document.getElementById('infoProcesamiento');

imgInput2.addEventListener('change', e => {
  const file = e.target.files[0];
  if (!file) return;

  const img = new Image();
  img.onload = () => {
    // Mostrar imagen original escalada al canvas
    ctxOriginal.clearRect(0, 0, canvasOriginal.width, canvasOriginal.height);
    ctxOriginal.drawImage(img, 0, 0, canvasOriginal.width, canvasOriginal.height);

    // Procesar imagen: subdividir y normalizar (ejemplo simple)
    const tileSize = 20;
    const tilesX = canvasOriginal.width / tileSize;
    const tilesY = canvasOriginal.height / tileSize;

    // Borra canvas procesado
    ctxProcesado.clearRect(0, 0, canvasProcesado.width, canvasProcesado.height);

    let valores = [];
    for(let y=0; y < tilesY; y++) {
      for(let x=0; x < tilesX; x++) {
        const imageData = ctxOriginal.getImageData(x*tileSize, y*tileSize, tileSize, tileSize);
        let suma = 0;
        for(let i=0; i < imageData.data.length; i += 4) {
          // Promedio RGB
          const r = imageData.data[i];
          const g = imageData.data[i+1];
          const b = imageData.data[i+2];
          suma += (r + g + b)/3;
        }
        const brilloPromedio = suma / (tileSize * tileSize);
        valores.push(brilloPromedio);

        // Dibujar en canvas procesado con brillo normalizado
        const brilloNorm = brilloPromedio / 255;
        ctxProcesado.fillStyle = `rgba(${Math.floor(brilloNorm*255)},0,${Math.floor((1-brilloNorm)*255)},0.8)`;
        ctxProcesado.fillRect(x*tileSize, y*tileSize, tileSize, tileSize);
      }
    }

    // Mostrar info simple
    infoProcesamiento.innerHTML = `
      <p>Subdividido en ${tilesX}x${tilesY} bloques.</p>
      <p>Valores promedio de brillo (normalizados):</p>
      <pre>${valores.map(v => (v/255).toFixed(2)).join(', ')}</pre>
    `;

    // Aquí podrías hacer más transformaciones y mostrar vectores (depende de tu modelo)
  };

  img.src = URL.createObjectURL(file);
});

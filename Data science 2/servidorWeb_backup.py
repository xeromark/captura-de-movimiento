import os
import sys
import json
import math
import base64
import time
import cv2
import numpy as np
import torch
import psycopg2
from flask import Flask, request, jsonify, Response
from dotenv import load_dotenv
from urllib.parse import urlparse

# Agregar paths correctos
ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT, ".."))
sys.path.insert(0, os.path.join(ROOT, "..", "signhandler"))

from signhandler.signer import FaceEmbeddingGenerator
from signhandler.comparator import SignatureComparator
from main import IntegratedSystem

# Cargar variables de entorno
load_dotenv()

# Configuración de base de datos desde variables de entorno
def get_db_params():
    """Obtiene parámetros de BD desde variables de entorno o valores por defecto"""
    database_url = os.getenv('DATABASE_URL')
    
    if database_url:
        # Parsear URL de conexión PostgreSQL
        parsed = urlparse(database_url)
        return {
            'dbname': parsed.path[1:],  # Remover el '/' inicial
            'user': parsed.username,
            'password': parsed.password,
            'host': parsed.hostname,
            'port': parsed.port or 5432
        }
    else:
        # Valores por defecto si no hay DATABASE_URL
        return {
            'dbname': os.getenv('DB_NAME', 'postgres'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'password'),
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '5432'))
        }

# Parámetros de BD globales
DB_PARAMS = get_db_params()

def crear_tabla_personas():
    """Crea la tabla de personas con nombres si no existe"""
    try:
        with psycopg2.connect(**DB_PARAMS) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS personas (
                        id SERIAL PRIMARY KEY,
                        nombre VARCHAR(255) NOT NULL,
                        firma TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
                print("✅ Tabla 'personas' verificada/creada en servidorWeb")
    except Exception as e:
        print(f"⚠️ Error creando tabla de personas en servidorWeb: {e}")

def probar_conexion_bd():
    """Prueba la conexión a la base de datos"""
    try:
        with psycopg2.connect(**DB_PARAMS) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()[0]
                print(f"✅ Conexión BD exitosa desde servidorWeb: {version}")
                return True
    except Exception as e:
        print(f"❌ Error conectando a BD desde servidorWeb: {e}")
        print(f"📋 Parámetros BD: {DB_PARAMS}")
        return False

# Inicializa el sistema principal con modelo ya preparado
sistema = IntegratedSystem(
    model_path=os.path.join(ROOT, "..", "signhandler", "model.pth"),
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Verificar y crear tabla si es necesario
print(f"📋 Conectando a BD desde servidorWeb: {DB_PARAMS['host']}:{DB_PARAMS['port']}")
if probar_conexion_bd():
    crear_tabla_personas()
else:
    print("⚠️ Continuando sin conexión a BD desde servidorWeb")

@app.route('/video_feed')
def video_feed():
    def gen():
        cap = None
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("❌ No se pudo acceder a la webcam")
                return
            
            print("✅ Webcam conectada para streaming")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Error al leer frame de la webcam")
                    break

                # Detección y dibujo
                gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                caras = sistema.detector_caras.detectMultiScale(gris, 1.1, 4)

                for (x, y, w, h) in caras:
                    try:
                        cara = frame[y:y+h, x:x+w]
                        firma = sistema.embedding_generator.generate_embedding(cara)
                        _, _, conocido, sim, nombre = sistema.comparar_firma_con_db(firma)
                        color = (0, 255, 0) if conocido else (0, 0, 255)
                        texto = f"{nombre} - {'Conocido' if conocido else 'Desconocido'} {sim:.1f}%"
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, texto, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    except Exception as e:
                        # Si hay error procesando la cara, solo dibujar rectángulo básico
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                        cv2.putText(frame, "Procesando...", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                _, buffer = cv2.imencode('.jpg', frame)
                frame_b64 = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_b64 + b'\r\n')
                
        except Exception as e:
            print(f"❌ Error en video_feed: {e}")
        finally:
            if cap:
                cap.release()
                print("📹 Webcam liberada")

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/guardar_firma', methods=['POST'])
def guardar_firma():
    """Guarda firma - compatible con index.html y nuevas funcionalidades"""
    
    # CREAR TABLA SI NO EXISTE (solución inmediata)
    try:
        with psycopg2.connect(**DB_PARAMS) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS personas (
                        id SERIAL PRIMARY KEY,
                        nombre VARCHAR(255) NOT NULL,
                        firma TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
                print("✅ Tabla 'personas' verificada/creada")
    except Exception as e:
        print(f"⚠️ Error creando tabla: {e}")
        return jsonify({"status": "error", "error": f"Error de base de datos: {e}"}), 500
    
    data = request.get_json()
    
    # Si viene con imagen base64 (modo original)
    img_b64 = data.get('imagen')
    nombre = data.get('nombre')
    
    # Si no viene imagen, capturar de webcam (nuevo modo)
    if not img_b64:
        if not nombre or nombre.strip() == "":
            return jsonify({"status": "error", "error": "Falta el nombre"}), 400
        
        try:
            # Capturar frame actual de la webcam como en guardar_firma_webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return jsonify({"status": "error", "error": "No se pudo acceder a la webcam"}), 500

            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return jsonify({"status": "error", "error": "No se pudo capturar imagen"}), 500

            # Convertir a escala de grises para detección
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            caras = sistema.detector_caras.detectMultiScale(gris, 1.1, 4)

            if len(caras) == 0:
                return jsonify({"status": "error", "error": "No se detectó rostro en la webcam"}), 422

            # Tomar la primera cara detectada
            x, y, w, h = caras[0]
            cara = frame[y:y+h, x:x+w]

            # Generar embedding facial
            embedding = sistema.embedding_generator.generate_embedding(cara)
            
            # Comparar con firmas existentes
            _, distance, is_known, similarity_percentage, nombre_existente = sistema.comparar_firma_con_db(embedding)

            # Guardar firma con el nombre proporcionado
            if sistema.insertar_firma(embedding, nombre.strip()):
                # Guardar imagen de la cara
                timestamp = time.time()
                status = "CONOCIDO" if is_known else "DESCONOCIDO"
                img_path = os.path.join(sistema.carpeta_capturas, f"cara_{nombre.strip()}_{similarity_percentage:.0f}pct_{timestamp}.jpg")
                cv2.imwrite(img_path, cara)
                
                print(f"✓ Firma guardada para '{nombre.strip()}' - Similitud: {similarity_percentage:.1f}%")
                print(f"📸 Imagen guardada en {img_path}")
                
                return jsonify({
                    "status": "ok", 
                    "mensaje": f"Firma guardada para '{nombre.strip()}'",
                    "firma": embedding,
                    "similitud": similarity_percentage,
                    "imagen_guardada": img_path,
                    "era_conocido": is_known,
                    "nombre_detectado": nombre_existente if is_known else "Nuevo"
                }), 200
            else:
                return jsonify({"status": "error", "error": "No se pudo guardar la firma en la base de datos"}), 500

        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 500
    
    # Modo original con imagen base64
    if not img_b64 or not nombre:
        return jsonify({"status": "error", "error": "Faltan datos"}), 400

    try:
        img_bytes = base64.b64decode(img_b64.split(",")[1])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        caras = sistema.detector_caras.detectMultiScale(gris, 1.1, 4)

        if len(caras) == 0:
            return jsonify({"status": "error", "error": "No se detectó rostro"}), 422

        x, y, w, h = caras[0]
        cara = img[y:y+h, x:x+w]

        embedding = sistema.embedding_generator.generate_embedding(cara)
        
        # Comparar con firmas existentes
        _, distance, is_known, similarity_percentage, nombre_existente = sistema.comparar_firma_con_db(embedding)

        if sistema.insertar_firma(embedding, nombre):
            # Guardar imagen de la cara
            timestamp = time.time()
            status = "CONOCIDO" if is_known else "DESCONOCIDO"
            img_path = os.path.join(sistema.carpeta_capturas, f"cara_{nombre.strip()}_{similarity_percentage:.0f}pct_{timestamp}.jpg")
            cv2.imwrite(img_path, cara)
            
            print(f"✓ Firma guardada para '{nombre.strip()}' - Similitud: {similarity_percentage:.1f}%")
            
            return jsonify({
                "status": "ok", 
                "firma": embedding,
                "similitud": similarity_percentage,
                "era_conocido": is_known,
                "nombre_detectado": nombre_existente if is_known else "Nuevo"
            }), 200
        else:
            return jsonify({"status": "error", "error": "No se pudo guardar firma"}), 500

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/comparar', methods=['POST'])
def comparar():
    data = request.get_json()
    img_b64 = data.get('imagen')

    if not img_b64:
        return jsonify({"error": "Falta imagen"}), 400

    try:
        img_bytes = base64.b64decode(img_b64.split(",")[1])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        caras = sistema.detector_caras.detectMultiScale(gris, 1.1, 4)

        if len(caras) == 0:
            return jsonify({"error": "No se detectó rostro"}), 422

        x, y, w, h = caras[0]
        cara = img[y:y+h, x:x+w]

        firma_b64 = sistema.embedding_generator.generate_embedding(cara)

        _, distance, conocido, similarity, nombre = sistema.comparar_firma_con_db(firma_b64)

        return jsonify({
            "firma": firma_b64, 
            "conocido": conocido, 
            "similaridad": similarity,
            "nombre": nombre
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generar_firma', methods=['POST'])
def generar_firma():
    data = request.get_json()
    img_b64 = data.get('imagen')

    if not img_b64:
        return jsonify({"error": "Falta imagen"}), 400

    try:
        img_bytes = base64.b64decode(img_b64.split(",")[1])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        caras = sistema.detector_caras.detectMultiScale(gris, 1.1, 4)

        if len(caras) == 0:
            return jsonify({"error": "No se detectó rostro"}), 422

        x, y, w, h = caras[0]
        cara = img[y:y+h, x:x+w]

        firma_b64 = sistema.embedding_generator.generate_embedding(cara)

        return jsonify({"firma": firma_b64}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/guardar_firma_webcam', methods=['POST'])
def guardar_firma_webcam():
    """Captura imagen de webcam y guarda firma con nombre como app.py"""
    data = request.get_json()
    nombre = data.get('nombre')

    if not nombre or nombre.strip() == "":
        return jsonify({"status": "error", "error": "Falta el nombre"}), 400

    try:
        # Capturar frame actual de la webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({"status": "error", "error": "No se pudo acceder a la webcam"}), 500

        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({"status": "error", "error": "No se pudo capturar imagen"}), 500

        # Convertir a escala de grises para detección
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        caras = sistema.detector_caras.detectMultiScale(gris, 1.1, 4)

        if len(caras) == 0:
            return jsonify({"status": "error", "error": "No se detectó rostro en la webcam"}), 422

        # Tomar la primera cara detectada
        x, y, w, h = caras[0]
        cara = frame[y:y+h, x:x+w]

        # Generar embedding facial como en app.py
        embedding = sistema.embedding_generator.generate_embedding(cara)
        
        # Comparar con firmas existentes para obtener similitud
        _, distance, is_known, similarity_percentage, nombre_existente = sistema.comparar_firma_con_db(embedding)

        # Guardar firma con el nombre proporcionado (como en app.py al presionar espacio)
        if sistema.insertar_firma(embedding, nombre.strip()):
            # Guardar imagen de la cara como en app.py
            timestamp = time.time()
            status = "CONOCIDO" if is_known else "DESCONOCIDO"
            img_path = os.path.join(sistema.carpeta_capturas, f"cara_{nombre.strip()}_{similarity_percentage:.0f}pct_{timestamp}.jpg")
            cv2.imwrite(img_path, cara)
            
            print(f"✓ Firma guardada para '{nombre.strip()}' - Similitud: {similarity_percentage:.1f}%")
            print(f"📸 Imagen guardada en {img_path}")
            
            return jsonify({
                "status": "ok", 
                "mensaje": f"Firma guardada para '{nombre.strip()}'",
                "firma": embedding,
                "similitud": similarity_percentage,
                "imagen_guardada": img_path,
                "era_conocido": is_known,
                "nombre_detectado": nombre_existente if is_known else "Nuevo"
            }), 200
        else:
            return jsonify({"status": "error", "error": "No se pudo guardar la firma en la base de datos"}), 500

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/')
def index():
    """Página principal con opciones"""
    return '''
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sistema de Reconocimiento Facial</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0; 
                padding: 20px; 
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .container { 
                max-width: 600px; 
                background: white; 
                padding: 40px; 
                border-radius: 15px; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.2); 
                text-align: center;
            }
            h1 {
                color: #333;
                margin-bottom: 30px;
            }
            .option {
                display: block;
                width: 100%;
                padding: 15px;
                margin: 15px 0;
                background: #007bff;
                color: white;
                text-decoration: none;
                border-radius: 8px;
                font-size: 18px;
                transition: background 0.3s;
            }
            .option:hover {
                background: #0056b3;
            }
            .status {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                margin: 20px 0;
                border-left: 4px solid #28a745;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 Sistema de Reconocimiento Facial</h1>
            <p>Elige cómo quieres usar el sistema:</p>
            
            <div class="status">
                <strong>Estado del Sistema:</strong><br>
                ✅ Modelo AI cargado<br>
                ✅ Base de datos PostgreSQL conectada<br>
                ✅ Nivel de exigencia: 85%
            </div>
            
            <a href="/index" class="option">
                🎨 Interfaz 3D Avanzada (index.html)
            </a>
            
            <a href="/simple" class="option">
                📹 Interfaz Simple de Captura
            </a>
            
            <p style="margin-top: 30px; color: #666;">
                <small>Puerto: 5000 | Tecnología: Redes Siamesas + PostgreSQL</small>
            </p>
        </div>
    </body>
    </html>
    '''

@app.route('/index')
def index_page():
    """Sirve el archivo index.html desde el servidor"""
    try:
        with open(os.path.join(ROOT, 'index.html'), 'r', encoding='utf-8') as f:
            html_content = f.read()
        # Actualizar la URL del video feed para que sea relativa
        html_content = html_content.replace('src="http://localhost:5000/video_feed"', 'src="/video_feed"')
        return html_content
    except FileNotFoundError:
        return "Error: archivo index.html no encontrado", 404

@app.route('/script.js')
def serve_script():
    """Sirve el archivo script.js"""
    try:
        with open(os.path.join(ROOT, 'script.js'), 'r', encoding='utf-8') as f:
            content = f.read()
        # Actualizar URLs en el JavaScript para que sean relativas
        content = content.replace('http://localhost:5000/', '/')
        return Response(content, mimetype='application/javascript')
    except FileNotFoundError:
        return "Error: archivo script.js no encontrado", 404

@app.route('/simple')
def simple_interface():
    """Interfaz simple de captura"""
    return '''
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Captura Simple - Reconocimiento Facial</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                background: #f0f0f0; 
                margin: 0; 
                padding: 20px; 
            }
            .container { 
                max-width: 800px; 
                margin: 0 auto; 
                background: white; 
                padding: 20px; 
                border-radius: 10px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
            }
            .video-container { 
                text-align: center; 
                margin: 20px 0; 
            }
            .video-stream { 
                max-width: 100%; 
                border: 2px solid #ddd; 
                border-radius: 8px; 
            }
            .controls { 
                display: flex; 
                gap: 10px; 
                margin: 20px 0; 
                align-items: center; 
            }
            input[type="text"] { 
                flex: 1; 
                padding: 10px; 
                border: 1px solid #ddd; 
                border-radius: 5px; 
            }
            button { 
                padding: 10px 20px; 
                background: #007bff; 
                color: white; 
                border: none; 
                border-radius: 5px; 
                cursor: pointer; 
            }
            button:hover { 
                background: #0056b3; 
            }
            .result { 
                margin: 20px 0; 
                padding: 15px; 
                border-radius: 5px; 
                background: #f8f9fa; 
                border-left: 4px solid #007bff; 
            }
            .back-link {
                display: inline-block;
                margin-bottom: 20px;
                color: #007bff;
                text-decoration: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <a href="/" class="back-link">← Volver al menú principal</a>
            
            <h1>🧠 Sistema de Reconocimiento Facial - Interfaz Simple</h1>
            <p>Captura y reconocimiento facial en tiempo real con IA</p>
            
            <div class="video-container">
                <img src="/video_feed" class="video-stream" alt="Video Stream">
            </div>
            
            <div class="controls">
                <input type="text" id="nombreInput" placeholder="Ingresa el nombre de la persona..." required>
                <button onclick="guardarFirmaWebcam()">📸 Capturar y Guardar Firma</button>
            </div>
            
            <div id="resultado" class="result" style="display: none;"></div>
        </div>
        
        <script>
            async function guardarFirmaWebcam() {
                const nombre = document.getElementById('nombreInput').value.trim();
                const resultado = document.getElementById('resultado');
                
                if (!nombre) {
                    alert('⚠️ Por favor, ingresa un nombre');
                    return;
                }
                
                const button = event.target;
                button.disabled = true;
                button.textContent = '🔄 Capturando...';
                
                try {
                    const response = await fetch('/guardar_firma_webcam', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ nombre: nombre })
                    });
                    
                    const data = await response.json();
                    
                    if (data.status === 'ok') {
                        resultado.innerHTML = `
                            <h3>✅ Firma guardada exitosamente</h3>
                            <p><strong>Nombre:</strong> ${nombre}</p>
                            <p><strong>Similitud con existentes:</strong> ${data.similitud.toFixed(1)}%</p>
                            <p><strong>Estado:</strong> ${data.era_conocido ? 'Era conocido como "' + data.nombre_detectado + '"' : 'Nueva persona'}</p>
                            <p><strong>Imagen guardada:</strong> ${data.imagen_guardada}</p>
                        `;
                        resultado.style.display = 'block';
                        resultado.style.borderLeftColor = '#28a745';
                        
                        // Limpiar input
                        document.getElementById('nombreInput').value = '';
                    } else {
                        resultado.innerHTML = `<h3>❌ Error</h3><p>${data.error}</p>`;
                        resultado.style.display = 'block';
                        resultado.style.borderLeftColor = '#dc3545';
                    }
                } catch (error) {
                    resultado.innerHTML = `<h3>❌ Error de conexión</h3><p>${error.message}</p>`;
                    resultado.style.display = 'block';
                    resultado.style.borderLeftColor = '#dc3545';
                } finally {
                    button.disabled = false;
                    button.textContent = '📸 Capturar y Guardar Firma';
                }
            }
        </script>
    </body>
    </html>
    '''

# Crear tabla de personas al iniciar el servidor
crear_tabla_personas()

# Probar conexión a la base de datos al iniciar el servidor
probar_conexion_bd()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
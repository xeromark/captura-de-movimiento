#!/usr/bin/env python3
"""
Servidor Web para Sistema de Reconocimiento Facial
Incluye interfaz web y conexión a PostgreSQL
"""

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

# Configuración de base de datos
def get_db_params():
    """Obtiene parámetros de BD desde variables de entorno"""
    database_url = os.getenv('DATABASE_URL')
    
    if database_url:
        parsed = urlparse(database_url)
        return {
            'dbname': parsed.path[1:],
            'user': parsed.username,
            'password': parsed.password,
            'host': parsed.hostname,
            'port': parsed.port or 5432
        }
    else:
        return {
            'dbname': os.getenv('DB_NAME', 'postgres'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'password'),
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '5432'))
        }

DB_PARAMS = get_db_params()

def crear_tabla_personas():
    """Crea la tabla de personas si no existe"""
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
                return True
    except Exception as e:
        print(f"⚠️ Error creando tabla: {e}")
        return False

# Inicializar Flask
app = Flask(__name__)

# Configurar CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/guardar_firma', methods=['OPTIONS'])
def handle_options():
    return '', 200

@app.route('/comparar', methods=['OPTIONS'])
def handle_options_comparar():
    return '', 200

# Inicializar sistema
print("🔄 Inicializando sistema...")
sistema = IntegratedSystem(
    model_path=os.path.join(ROOT, "..", "signhandler", "model.pth"),
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Crear tabla
print("🔄 Creando tabla de base de datos...")
crear_tabla_personas()

@app.route('/video_feed')
def video_feed():
    def gen():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ No se pudo acceder a la webcam")
            return
            
        while True:
            ret, frame = cap.read()
            if not ret:
                break

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
                except:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        cap.release()

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/guardar_firma', methods=['POST'])
def guardar_firma():
    """Guarda firma desde webcam"""
    # Asegurar que la tabla existe
    crear_tabla_personas()
    
    data = request.get_json()
    nombre = data.get('nombre')

    if not nombre or nombre.strip() == "":
        return jsonify({"status": "error", "error": "Falta el nombre"}), 400

    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({"status": "error", "error": "No se pudo acceder a la webcam"}), 500

        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({"status": "error", "error": "No se pudo capturar imagen"}), 500

        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        caras = sistema.detector_caras.detectMultiScale(gris, 1.1, 4)

        if len(caras) == 0:
            return jsonify({"status": "error", "error": "No se detectó rostro"}), 422

        x, y, w, h = caras[0]
        cara = frame[y:y+h, x:x+w]
        embedding = sistema.embedding_generator.generate_embedding(cara)
        
        _, distance, is_known, similarity_percentage, nombre_existente = sistema.comparar_firma_con_db(embedding)

        if sistema.insertar_firma(embedding, nombre.strip()):
            timestamp = time.time()
            img_path = os.path.join(sistema.carpeta_capturas, f"cara_{nombre.strip()}_{similarity_percentage:.0f}pct_{timestamp}.jpg")
            cv2.imwrite(img_path, cara)
            
            print(f"✓ Firma guardada para '{nombre.strip()}' - Similitud: {similarity_percentage:.1f}%")
            
            return jsonify({
                "status": "ok", 
                "mensaje": f"Firma guardada para '{nombre.strip()}'",
                "firma": embedding,
                "similitud": similarity_percentage,
                "era_conocido": is_known,
                "nombre_detectado": nombre_existente if is_known else "Nuevo"
            }), 200
        else:
            return jsonify({"status": "error", "error": "No se pudo guardar la firma"}), 500

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sistema de Reconocimiento Facial</title>
        <style>
            body { font-family: Arial; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .video { max-width: 100%; border: 2px solid #ddd; }
            .controls { margin: 20px 0; }
            input { padding: 10px; margin: 10px; width: 200px; }
            button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 Sistema de Reconocimiento Facial</h1>
            <img src="/video_feed" class="video">
            <div class="controls">
                <input type="text" id="nombre" placeholder="Nombre de la persona">
                <button onclick="guardarFirma()">Guardar Firma</button>
            </div>
            <div id="resultado"></div>
        </div>
        
        <script>
            async function guardarFirma() {
                const nombre = document.getElementById('nombre').value.trim();
                if (!nombre) {
                    alert('Ingresa un nombre');
                    return;
                }
                
                try {
                    const response = await fetch('/guardar_firma', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ nombre })
                    });
                    
                    const data = await response.json();
                    
                    if (data.status === 'ok') {
                        document.getElementById('resultado').innerHTML = 
                            `<h3>✅ Éxito</h3><p>Firma guardada para: ${nombre}</p><p>Similitud: ${data.similitud.toFixed(1)}%</p>`;
                        document.getElementById('nombre').value = '';
                    } else {
                        document.getElementById('resultado').innerHTML = 
                            `<h3>❌ Error</h3><p>${data.error}</p>`;
                    }
                } catch (error) {
                    document.getElementById('resultado').innerHTML = 
                        `<h3>❌ Error</h3><p>${error.message}</p>`;
                }
            }
        </script>
    </body>
    </html>
    '''

if __name__ == "__main__":
    print("🚀 Iniciando servidor en http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)

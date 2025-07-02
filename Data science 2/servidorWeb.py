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
import threading
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
from main import IntegratedSystem, probar_conexion_bd

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

@app.route('/status')
def system_status():
    """Verifica el estado del sistema"""
    try:
        # Verificar BD
        bd_ok = probar_conexion_bd()
        
        # Verificar cámara
        frame = tomar_foto_segura()
        camera_ok = frame is not None
        
        # Verificar modelo
        model_ok = os.path.exists(sistema.model_path) if hasattr(sistema, 'model_path') else False
        
        return jsonify({
            "status": "ok",
            "database": "ok" if bd_ok else "error",
            "camera": "ok" if camera_ok else "error", 
            "model": "ok" if model_ok else "error",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/camera_test')
def camera_test():
    """Prueba la cámara y devuelve una imagen"""
    try:
        frame = tomar_foto_segura()
        if frame is None:
            return jsonify({"status": "error", "error": "No se pudo acceder a la cámara"}), 500
        
        # Convertir a base64 para mostrar en el navegador
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "status": "ok",
            "image_shape": frame.shape,
            "image_data": f"data:image/jpeg;base64,{img_base64}"
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

# Inicializar sistema
print("🔄 Inicializando sistema...")
sistema = IntegratedSystem(
    model_path=os.path.join(ROOT, "..", "signhandler", "model.pth"),
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Crear tabla
print("🔄 Creando tabla de base de datos...")
crear_tabla_personas()

# Estado global para controlar acceso a la cámara
camera_lock = threading.Lock()

def tomar_foto_segura():
    """Toma una foto de forma segura sin conflictos"""
    with camera_lock:
        cap = None
        try:
            # Intentar múltiples índices de cámara
            for cam_index in range(3):
                try:
                    cap = cv2.VideoCapture(cam_index)
                    if cap.isOpened():
                        # Configurar resolución y FPS para mejor rendimiento
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        
                        # Dar tiempo a la cámara para inicializar
                        time.sleep(0.3)
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            cap.release()
                            return frame
                    if cap:
                        cap.release()
                except:
                    if cap:
                        cap.release()
                    continue
            return None
        except Exception as e:
            print(f"❌ Error tomando foto: {e}")
            if cap:
                cap.release()
            return None

@app.route('/video_feed')
def video_feed():
    """Video feed que toma fotos puntuales sin conflictos"""
    def gen():
        while True:
            try:
                frame = tomar_foto_segura()
                
                if frame is None:
                    # Frame de error si no puede acceder a la cámara
                    error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(error_frame, "Camera Error", (200, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(error_frame, "Check if camera is free", (150, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    _, buffer = cv2.imencode('.jpg', error_frame)
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    time.sleep(2)  # Pausa más larga para errores
                    continue

                # Procesar frame para detección facial
                try:
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
                            cv2.putText(frame, "Procesando...", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # Agregar timestamp
                    timestamp = time.strftime("%H:%M:%S")
                    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                except Exception as e:
                    print(f"⚠️ Error procesando frame: {e}")

                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
                # Pausa entre capturas - más tiempo para reducir carga
                time.sleep(0.5)  # 2 FPS para el preview
                
            except Exception as e:
                print(f"❌ Error en video_feed: {e}")
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, "Stream Error", (220, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                _, buffer = cv2.imencode('.jpg', error_frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(2)

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/guardar_firma', methods=['POST'])
def guardar_firma():
    """Guarda firma tomando foto puntual como app.py"""
    # Asegurar que la tabla existe
    crear_tabla_personas()
    
    data = request.get_json()
    nombre = data.get('nombre')

    if not nombre or nombre.strip() == "":
        return jsonify({"status": "error", "error": "Falta el nombre"}), 400

    try:
        print(f"📸 Intentando capturar imagen para '{nombre}'...")
        
        # Usar la función segura de captura
        frame = tomar_foto_segura()
        
        if frame is None:
            return jsonify({
                "status": "error", 
                "error": "No se pudo capturar imagen. Cámara ocupada por otra aplicación."
            }), 500

        print(f"📸 Imagen capturada: {frame.shape}")

        # Procesar imagen como en app.py
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        caras = sistema.detector_caras.detectMultiScale(gris, 1.1, 4)

        if len(caras) == 0:
            return jsonify({
                "status": "error", 
                "error": "No se detectó rostro en la imagen. Posiciónate frente a la cámara."
            }), 422

        # Tomar la primera cara detectada (como en app.py)
        x, y, w, h = caras[0]
        cara = frame[y:y+h, x:x+w]
        print(f"👤 Cara detectada: {cara.shape} en posición ({x},{y},{w},{h})")

        # Generar embedding facial directamente como en app.py
        embedding = sistema.embedding_generator.generate_embedding(cara)
        
        # Comparar con firmas existentes como en app.py
        _, distance, is_known, similarity_percentage, nombre_existente = sistema.comparar_firma_con_db(embedding)

        # Insertar firma como en app.py
        if sistema.insertar_firma(embedding, nombre.strip()):
            print(f"✓ Firma guardada para '{nombre.strip()}' - Similitud: {similarity_percentage:.1f}%")
            print(f"� Solo firma guardada en PostgreSQL (sin imagen física)")
            
            return jsonify({
                "status": "ok", 
                "mensaje": f"Firma guardada para '{nombre.strip()}'",
                "firma": embedding,
                "similitud": similarity_percentage,
                "era_conocido": is_known,
                "nombre_detectado": nombre_existente if is_known else "Nuevo",
                "posicion_cara": f"({x},{y},{w},{h})"
            }), 200
        else:
            return jsonify({
                "status": "error", 
                "error": "No se pudo guardar la firma en la base de datos"
            }), 500

    except Exception as e:
        print(f"❌ Error en guardar_firma: {e}")
        return jsonify({
            "status": "error", 
            "error": f"Error interno: {str(e)}"
        }), 500

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sistema de Reconocimiento Facial</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 20px; 
                background-color: #f5f5f5;
            }
            .container { 
                max-width: 1000px; 
                margin: 0 auto; 
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
                color: #333;
            }
            .video-container {
                text-align: center;
                margin: 20px 0;
            }
            .video { 
                max-width: 100%; 
                border: 3px solid #ddd; 
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            }
            .controls { 
                margin: 20px 0; 
                text-align: center;
            }
            input { 
                padding: 12px; 
                margin: 10px; 
                width: 250px; 
                border: 2px solid #ddd;
                border-radius: 5px;
                font-size: 16px;
            }
            button { 
                padding: 12px 25px; 
                background: #007bff; 
                color: white; 
                border: none; 
                cursor: pointer;
                border-radius: 5px;
                font-size: 16px;
                margin: 10px;
                transition: background-color 0.3s;
            }
            button:hover {
                background: #0056b3;
            }
            button:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
            .status {
                margin: 20px 0;
                padding: 15px;
                border-radius: 5px;
                text-align: center;
            }
            .status.success {
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            .status.error {
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            .status.info {
                background: #d1ecf1;
                color: #0c5460;
                border: 1px solid #bee5eb;
            }
            .system-status {
                display: flex;
                justify-content: space-around;
                margin: 20px 0;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 5px;
            }
            .status-item {
                text-align: center;
            }
            .status-indicator {
                width: 20px;
                height: 20px;
                border-radius: 50%;
                margin: 0 auto 5px;
            }
            .status-ok { background: #28a745; }
            .status-error { background: #dc3545; }
            .loading {
                display: none;
                text-align: center;
                margin: 20px 0;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #3498db;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 2s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 Sistema de Reconocimiento Facial</h1>
                <p>Captura y reconocimiento de firmas faciales con IA</p>
            </div>
            
            <div class="system-status" id="systemStatus">
                <div class="status-item">
                    <div class="status-indicator status-error" id="dbStatus"></div>
                    <small>Base de Datos</small>
                </div>
                <div class="status-item">
                    <div class="status-indicator status-error" id="cameraStatus"></div>
                    <small>Cámara</small>
                </div>
                <div class="status-item">
                    <div class="status-indicator status-error" id="modelStatus"></div>
                    <small>Modelo IA</small>
                </div>
            </div>
            
            <div class="video-container">
                <img src="/video_feed" class="video" id="videoFeed" alt="Video en vivo">
            </div>
            
            <div class="controls">
                <input type="text" id="nombre" placeholder="Nombre de la persona" maxlength="50">
                <br>
                <button onclick="guardarFirma()" id="guardarBtn">📸 Guardar Firma</button>
                <button onclick="testCamera()" id="testBtn">🔧 Probar Cámara</button>
                <button onclick="checkStatus()" id="statusBtn">⚡ Estado Sistema</button>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Procesando...</p>
            </div>
            
            <div id="resultado"></div>
        </div>
        
        <script>
            let isProcessing = false;
            
            // Verificar estado del sistema al cargar
            window.onload = function() {
                checkStatus();
            };
            
            function showLoading(show) {
                document.getElementById('loading').style.display = show ? 'block' : 'none';
                document.getElementById('guardarBtn').disabled = show;
                isProcessing = show;
            }
            
            function showMessage(message, type, details = '') {
                const resultado = document.getElementById('resultado');
                const icon = type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️';
                resultado.innerHTML = `
                    <div class="status ${type}">
                        <h3>${icon} ${type === 'success' ? 'Éxito' : type === 'error' ? 'Error' : 'Información'}</h3>
                        <p>${message}</p>
                        ${details ? `<small>${details}</small>` : ''}
                    </div>
                `;
            }
            
            async function checkStatus() {
                try {
                    const response = await fetch('/status');
                    const data = await response.json();
                    
                    document.getElementById('dbStatus').className = 
                        'status-indicator ' + (data.database === 'ok' ? 'status-ok' : 'status-error');
                    document.getElementById('cameraStatus').className = 
                        'status-indicator ' + (data.camera === 'ok' ? 'status-ok' : 'status-error');
                    document.getElementById('modelStatus').className = 
                        'status-indicator ' + (data.model === 'ok' ? 'status-ok' : 'status-error');
                        
                    showMessage(`Estado actualizado: ${data.timestamp}`, 'info');
                } catch (error) {
                    showMessage('Error verificando estado del sistema', 'error', error.message);
                }
            }
            
            async function testCamera() {
                showLoading(true);
                try {
                    const response = await fetch('/camera_test');
                    const data = await response.json();
                    
                    if (data.status === 'ok') {
                        showMessage('Cámara funcionando correctamente', 'success', 
                            `Resolución: ${data.image_shape[1]}x${data.image_shape[0]}`);
                    } else {
                        showMessage('Error en la cámara', 'error', data.error);
                    }
                } catch (error) {
                    showMessage('Error probando cámara', 'error', error.message);
                } finally {
                    showLoading(false);
                }
            }
            
            async function guardarFirma() {
                if (isProcessing) return;
                
                const nombre = document.getElementById('nombre').value.trim();
                if (!nombre) {
                    showMessage('Por favor ingresa un nombre', 'error');
                    return;
                }
                
                showLoading(true);
                
                try {
                    const response = await fetch('/guardar_firma', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ nombre })
                    });
                    
                    const data = await response.json();
                    
                    if (data.status === 'ok') {
                        showMessage(
                            `Firma guardada exitosamente para: ${nombre}`, 
                            'success',
                            `Similitud: ${data.similitud.toFixed(1)}% | ${data.era_conocido ? 'Persona conocida' : 'Nueva persona'}`
                        );
                        document.getElementById('nombre').value = '';
                        
                        // Actualizar estado del sistema
                        setTimeout(checkStatus, 1000);
                    } else {
                        showMessage('Error al guardar firma', 'error', data.error);
                    }
                } catch (error) {
                    showMessage('Error de conexión', 'error', error.message);
                } finally {
                    showLoading(false);
                }
            }
            
            // Manejo de errores en el video feed
            document.getElementById('videoFeed').onerror = function() {
                this.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480"><rect width="100%" height="100%" fill="%23f0f0f0"/><text x="50%" y="50%" text-anchor="middle" dy=".3em" font-family="Arial" font-size="24" fill="%23666">Video no disponible</text></svg>';
            };
            
            // Permitir envío con Enter
            document.getElementById('nombre').addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && !isProcessing) {
                    guardarFirma();
                }
            });
        </script>
    </body>
    </html>
    '''

if __name__ == "__main__":
    print("🚀 Iniciando servidor en http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)

#!/usr/bin/env python3
"""
Servidor Web para Sistema de Reconocimiento Facial
Incluye interfaz web con animaciones y conexión a PostgreSQL
"""

import os
import sys
import time
import base64
import math
import psycopg2
import cv2
import numpy as np
import torch
from flask import Flask, render_template_string, request, jsonify, Response
from dotenv import load_dotenv
from urllib.parse import urlparse

# Configurar paths
ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT, "signhandler"))

# Importar componentes
from signhandler.comparator import SignatureComparator
from signhandler.signer import FaceEmbeddingGenerator

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

def crear_tabla_firmas():
    """Crea la tabla de firmas si no existe"""
    try:
        with psycopg2.connect(**DB_PARAMS) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS firmas (
                        id SERIAL PRIMARY KEY,
                        nombre VARCHAR(100) NOT NULL,
                        firma TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
    except Exception as e:
        print(f"Error creando tabla: {e}")

# Configuración global
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = os.path.join(ROOT, "signhandler", "model.pth")

app = Flask(__name__)

# Template HTML con animaciones mejoradas
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sistema de Reconocimiento Facial</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            color: white;
        }
        
        .header {
            background: rgba(0,0,0,0.2);
            padding: 20px;
            text-align: center;
            border-bottom: 2px solid rgba(255,255,255,0.1);
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .status-bar {
            background: rgba(0,0,0,0.3);
            padding: 15px;
            text-align: center;
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
        }
        
        .status-item {
            margin: 5px;
            padding: 10px 20px;
            background: rgba(255,255,255,0.1);
            border-radius: 20px;
            backdrop-filter: blur(10px);
        }
        
        .container {
            flex: 1;
            display: flex;
            padding: 20px;
            gap: 20px;
            flex-wrap: wrap;
        }
        
        .left-panel, .right-panel {
            flex: 1;
            min-width: 300px;
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .panel-title {
            font-size: 1.5em;
            margin-bottom: 20px;
            text-align: center;
            color: #fff;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        }
        
        .video-container {
            background: rgba(0,0,0,0.3);
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            text-align: center;
        }
        
        .video-stream {
            max-width: 100%;
            border-radius: 10px;
            border: 2px solid rgba(255,255,255,0.2);
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
        }
        
        input, textarea, button {
            width: 100%;
            padding: 12px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            background: rgba(255,255,255,0.9);
            color: #333;
        }
        
        button {
            background: linear-gradient(45deg, #28a745, #20c997);
            color: white;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        
        .result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 10px;
            background: rgba(255,255,255,0.1);
            border-left: 4px solid #28a745;
        }
        
        .firmas-list {
            max-height: 400px;
            overflow-y: auto;
            background: rgba(0,0,0,0.2);
            border-radius: 10px;
            padding: 15px;
        }
        
        .firma-item {
            background: rgba(255,255,255,0.1);
            margin: 10px 0;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #17a2b8;
            animation: slideIn 0.5s ease;
        }
        
        .neural-animation {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: -1;
            opacity: 0.1;
        }
        
        .neural-node {
            position: absolute;
            width: 4px;
            height: 4px;
            background: white;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        .processing {
            position: relative;
            overflow: hidden;
        }
        
        .processing::after {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            animation: processing 1.5s infinite;
        }
        
        .similarity-bar {
            width: 100%;
            height: 20px;
            background: rgba(255,255,255,0.2);
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .similarity-fill {
            height: 100%;
            background: linear-gradient(90deg, #dc3545, #ffc107, #28a745);
            border-radius: 10px;
            transition: width 0.5s ease;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        
        .stat-card {
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #28a745;
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 0.3; transform: scale(1); }
            50% { opacity: 1; transform: scale(1.5); }
        }
        
        @keyframes processing {
            0% { left: -100%; }
            100% { left: 100%; }
        }
    </style>
</head>
<body>
    <div class="neural-animation" id="neuralAnimation"></div>
    
    <div class="header">
        <h1>🧠 Sistema de Reconocimiento Facial</h1>
        <p>Tecnología de Deep Learning con Redes Siamesas</p>
    </div>
    
    <div class="status-bar">
        <div class="status-item">
            <strong>Estado:</strong> Activo
        </div>
        <div class="status-item">
            <strong>Modelo:</strong> {{ device_info }}
        </div>
        <div class="status-item">
            <strong>Precisión:</strong> 85%+ Requerido
        </div>
        <div class="status-item">
            <strong>Firmas:</strong> {{ total_firmas }}
        </div>
    </div>
    
    <div class="container">
        <div class="left-panel">
            <h2 class="panel-title">📹 Video en Tiempo Real</h2>
            <div class="video-container">
                <img src="/video_feed" class="video-stream" alt="Video Stream">
                <p>🎯 Detección facial en tiempo real con IA</p>
            </div>
            
            <h2 class="panel-title">🔍 Comparar Firma</h2>
            <form id="compareForm">
                <div class="form-group">
                    <label for="firma1">Primera Firma (Base64):</label>
                    <textarea id="firma1" rows="3" placeholder="Pega aquí la primera firma..."></textarea>
                </div>
                <div class="form-group">
                    <label for="firma2">Segunda Firma (Base64):</label>
                    <textarea id="firma2" rows="3" placeholder="Pega aquí la segunda firma..."></textarea>
                </div>
                <button type="submit">🚀 Analizar Similitud</button>
            </form>
            
            <div id="compareResult" class="result" style="display: none;">
                <h3>📊 Resultado del Análisis</h3>
                <div id="similarityDetails"></div>
            </div>
        </div>
        
        <div class="right-panel">
            <h2 class="panel-title">👥 Firmas Registradas</h2>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number">{{ total_firmas }}</div>
                    <div>Total Registradas</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">85%</div>
                    <div>Precisión Mínima</div>
                </div>
            </div>
            
            <div class="firmas-list">
                {% for firma in firmas %}
                <div class="firma-item">
                    <div><strong>👤 {{ firma.nombre }}</strong></div>
                    <div>📅 Registrada: {{ firma.created_at.strftime('%d/%m/%Y %H:%M') if firma.created_at else 'Sin fecha' }}</div>
                    <div>🆔 ID: {{ firma.id }}</div>
                </div>
                {% endfor %}
            </div>
        </div>
    </div>
    
    <script>
        // Animación de nodos neurales
        function createNeuralAnimation() {
            const container = document.getElementById('neuralAnimation');
            const nodeCount = 50;
            
            for (let i = 0; i < nodeCount; i++) {
                const node = document.createElement('div');
                node.className = 'neural-node';
                node.style.left = Math.random() * 100 + '%';
                node.style.top = Math.random() * 100 + '%';
                node.style.animationDelay = Math.random() * 2 + 's';
                container.appendChild(node);
            }
        }
        
        // Función para mostrar barra de similitud
        function showSimilarityBar(percentage) {
            return `
                <div class="similarity-bar">
                    <div class="similarity-fill" style="width: ${percentage}%"></div>
                </div>
                <p><strong>Similitud:</strong> ${percentage.toFixed(1)}%</p>
            `;
        }
        
        // Manejar formulario de comparación
        document.getElementById('compareForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const firma1 = document.getElementById('firma1').value.trim();
            const firma2 = document.getElementById('firma2').value.trim();
            const resultDiv = document.getElementById('compareResult');
            const detailsDiv = document.getElementById('similarityDetails');
            
            if (!firma1 || !firma2) {
                alert('⚠️ Por favor, ingresa ambas firmas');
                return;
            }
            
            const submitBtn = e.target.querySelector('button');
            submitBtn.classList.add('processing');
            submitBtn.textContent = '🔄 Procesando...';
            submitBtn.disabled = true;
            
            try {
                const response = await fetch('/comparar', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ firma1: firma1, firma2: firma2 })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    const percentage = data.similarity_percentage;
                    const isMatch = percentage >= 85;
                    
                    detailsDiv.innerHTML = `
                        ${showSimilarityBar(percentage)}
                        <p><strong>Distancia:</strong> ${data.distance.toFixed(4)}</p>
                        <p><strong>Estado:</strong> 
                            <span style="color: ${isMatch ? '#28a745' : '#dc3545'}; font-weight: bold;">
                                ${isMatch ? '✅ COINCIDENCIA' : '❌ NO COINCIDE'}
                            </span>
                        </p>
                    `;
                    
                    resultDiv.style.display = 'block';
                    resultDiv.style.borderLeftColor = isMatch ? '#28a745' : '#dc3545';
                } else {
                    detailsDiv.innerHTML = `<p style="color: #dc3545;">❌ Error: ${data.message}</p>`;
                    resultDiv.style.display = 'block';
                }
            } catch (error) {
                detailsDiv.innerHTML = `<p style="color: #dc3545;">❌ Error: ${error.message}</p>`;
                resultDiv.style.display = 'block';
            } finally {
                submitBtn.classList.remove('processing');
                submitBtn.textContent = '🚀 Analizar Similitud';
                submitBtn.disabled = false;
            }
        });
        
        // Inicializar animaciones
        createNeuralAnimation();
    </script>
</body>
</html>
"""

class FaceRecognitionSystem:
    def __init__(self):
        self.device = DEVICE
        self.model_path = MODEL_PATH
        self.min_similarity_percentage = 85.0
        
        # Inicializar modelo
        try:
            self.comparator = SignatureComparator(self.model_path, device=self.device)
            self.embedding_generator = FaceEmbeddingGenerator(self.model_path, device=self.device)
            self.detector_caras = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            print(f"✅ Modelo cargado en {self.device}")
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            self.comparator = None
            self.embedding_generator = None
            self.detector_caras = None
        
        # Crear tabla en BD
        crear_tabla_firmas()
    
    def obtener_firmas_db(self):
        """Obtiene todas las firmas de la base de datos"""
        try:
            with psycopg2.connect(**DB_PARAMS) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id, nombre, firma, created_at 
                        FROM firmas 
                        WHERE firma IS NOT NULL AND LENGTH(firma) > 0
                        ORDER BY created_at DESC
                    """)
                    resultados = cur.fetchall()
                    
                    firmas = []
                    for id_firma, nombre, firma, created_at in resultados:
                        try:
                            base64.b64decode(firma)  # Validar base64
                            firmas.append({
                                'id': id_firma,
                                'nombre': nombre,
                                'firma': firma,
                                'created_at': created_at
                            })
                        except Exception:
                            continue
                    
                    return firmas
        except Exception as e:
            print(f"Error obteniendo firmas: {e}")
            return []
    
    def comparar_firmas(self, firma1, firma2):
        """Compara dos firmas y devuelve distancia y similitud"""
        if not self.comparator:
            return None, 0.0, False
        
        try:
            # Validar base64
            base64.b64decode(firma1)
            base64.b64decode(firma2)
            
            # Comparar usando lógica Discord
            distance = self.comparator.compare_with_discord_logic(firma1, firma2)
            
            # Calcular porcentaje de similitud
            similarity_percentage = max(0, 100 * math.exp(-distance / 2))
            
            # Determinar si es una coincidencia
            is_match = similarity_percentage >= self.min_similarity_percentage
            
            return distance, similarity_percentage, is_match
            
        except Exception as e:
            print(f"Error comparando firmas: {e}")
            return None, 0.0, False

    def comparar_firma_con_db(self, firma_nueva):
        """Compara una firma con todas las de la BD"""
        firmas_db = self.obtener_firmas_db()
        if not firmas_db:
            return None, float('inf'), False, 0.0, "Desconocido"
        
        min_distance = float('inf')
        nombre_mas_similar = "Desconocido"
        
        for item in firmas_db:
            try:
                firma_db = item['firma']
                nombre_db = item['nombre']
                
                distance = self.comparator.compare_with_discord_logic(firma_nueva, firma_db)
                
                if distance < min_distance:
                    min_distance = distance
                    nombre_mas_similar = nombre_db
                    
            except Exception:
                continue
        
        # Calcular porcentaje de similitud
        if min_distance == float('inf'):
            similarity_percentage = 0.0
        else:
            similarity_percentage = max(0, 100 * math.exp(-min_distance / 2))
        
        # Determinar si es conocido
        conocido = similarity_percentage >= self.min_similarity_percentage
        
        return None, min_distance, conocido, similarity_percentage, nombre_mas_similar

# Inicializar sistema
sistema = FaceRecognitionSystem()

@app.route('/')
def index():
    """Página principal"""
    firmas = sistema.obtener_firmas_db()
    device_info = f"{sistema.device.upper()} ({'GPU' if sistema.device == 'cuda' else 'CPU'})"
    
    return render_template_string(HTML_TEMPLATE, 
                                firmas=firmas, 
                                total_firmas=len(firmas),
                                device_info=device_info)

@app.route('/video_feed')
def video_feed():
    """Stream de video con detección facial"""
    def gen():
        if not sistema.detector_caras or not sistema.embedding_generator:
            return
            
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detección y dibujo
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            caras = sistema.detector_caras.detectMultiScale(gris, 1.1, 4)

            for (x, y, w, h) in caras:
                cara = frame[y:y+h, x:x+w]
                firma = sistema.embedding_generator.generate_embedding(cara)
                _, _, conocido, sim, nombre = sistema.comparar_firma_con_db(firma)
                color = (0, 255, 0) if conocido else (0, 0, 255)
                if conocido:
                    texto = f"{nombre} {sim:.1f}%"
                else:
                    texto = f"Desconocido {sim:.1f}%"
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, texto, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_b64 = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_b64 + b'\r\n')
        
        cap.release()

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/comparar', methods=['POST'])
def comparar():
    """API para comparar dos firmas"""
    try:
        data = request.get_json()
        firma1 = data.get('firma1', '').strip()
        firma2 = data.get('firma2', '').strip()
        
        if not firma1 or not firma2:
            return jsonify({
                'success': False,
                'message': 'Se requieren ambas firmas'
            })
        
        distance, similarity_percentage, is_match = sistema.comparar_firmas(firma1, firma2)
        
        if distance is None:
            return jsonify({
                'success': False,
                'message': 'Error al procesar las firmas'
            })
        
        return jsonify({
            'success': True,
            'distance': distance,
            'similarity_percentage': similarity_percentage,
            'is_match': is_match,
            'threshold': sistema.min_similarity_percentage
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error interno: {str(e)}'
        })

@app.route('/stats')
def stats():
    """API para estadísticas del sistema"""
    try:
        firmas = sistema.obtener_firmas_db()
        return jsonify({
            'total_firmas': len(firmas),
            'precision_minima': sistema.min_similarity_percentage,
            'device': sistema.device,
            'modelo_cargado': sistema.comparator is not None
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        })

if __name__ == '__main__':
    print("🚀 Iniciando Servidor Web de Reconocimiento Facial")
    print(f"📊 Firmas en BD: {len(sistema.obtener_firmas_db())}")
    print(f"🧠 Modelo: {sistema.device.upper()}")
    print(f"🎯 Precisión mínima: {sistema.min_similarity_percentage}%")
    print("🌐 Accede a: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

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

        _, distance, conocido, similarity = sistema.comparar_firma_con_db(firma_b64)

        return jsonify({"firma": firma_b64, "conocido": conocido, "similaridad": similarity}), 200

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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
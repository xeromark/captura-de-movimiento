#!/usr/bin/env python3
"""
Sistema Integrado de Captura de Movimiento y Procesamiento de Firmas
Combina funcionalidades de captura de cámara y procesamiento de firmas en un solo ejecutable.
"""

import argparse
import threading
import sys
import os
import time
import base64
import tempfile
import cv2
import queue

# Configuración de paths
ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT, "container"))
sys.path.insert(0, os.path.join(ROOT, "signhandler"))

# Importaciones de librerías externas
import torch
import psycopg2
from flask import Flask, request, jsonify
import requests
from dotenv import load_dotenv
from urllib.parse import urlparse

# Cargar variables de entorno
load_dotenv()

# Importar funciones específicas de los módulos como bloques de construcción
from signhandler.siamese_network import SiameseNetwork
from signhandler.signer import sign_image, generate_keys, capture_square_photo
from signhandler.comparator import SignatureComparator
from container.external_cam import capture_from_ip_camera, scan_network_for_cameras
from container.camera_photo import capturar_movimiento
from container.sender import enviar_imagen_post, enviar_imagenes_a_ip, monitorear_y_enviar

# Configuración global
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = os.path.join(ROOT, "signhandler", "model.pth")

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
            'dbname': os.getenv('DB_NAME', 'signatures'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres'),
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '5432'))
        }

# Parámetros de BD globales
db_params = get_db_params()

def crear_tabla_firmas():
    """Crea la tabla de firmas si no existe"""
    try:
        with psycopg2.connect(**db_params) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS firmas (
                        id SERIAL PRIMARY KEY,
                        firma TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
                print("✅ Tabla 'firmas' verificada/creada")
    except Exception as e:
        print(f"⚠️ Error creando tabla de firmas: {e}")

def insertar_firma_bd(firma):
    """Inserta una firma en la base de datos"""
    try:
        with psycopg2.connect(**db_params) as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO firmas (firma) VALUES (%s)", (firma,))
                conn.commit()
                return True
    except Exception as e:
        print(f"❌ Error insertando firma en BD: {e}")
        return False

def probar_conexion_bd():
    """Prueba la conexión a la base de datos"""
    try:
        with psycopg2.connect(**db_params) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()[0]
                print(f"✅ Conexión BD exitosa: {version}")
                return True
    except Exception as e:
        print(f"❌ Error conectando a BD: {e}")
        print(f"📋 Parámetros BD: {db_params}")
        return False

class CapturaYFirma:
    """Clase principal que integra captura de cámara y procesamiento de firmas"""
    
    def __init__(self, carpeta_capturas="capturas", ip_destino="192.168.1.100", puerto=5000):
        self.carpeta_capturas = carpeta_capturas
        self.ip_destino = ip_destino
        self.puerto = puerto
        self.running = False
        self.contador = 0
        
        # Crear carpeta de capturas
        os.makedirs(carpeta_capturas, exist_ok=True)
        
        # Crear tabla de firmas si no existe
        crear_tabla_firmas()
        
        # Inicializar componentes de procesamiento
        self.detector_caras = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.comparator = SignatureComparator(MODEL_PATH, device=DEVICE)
        self.priv_key, self.pub_key = generate_keys()
        
        # Configurar Flask app
        self.app = Flask(__name__)
        self.setup_flask_routes()
    
    def setup_flask_routes(self):
        """Configura las rutas del servidor Flask"""
        
        @self.app.route('/comparar', methods=['POST'])
        def comparar_firma():
            data = request.get_json(silent=True) or {}
            foto_b64 = data.get('foto')
            if not foto_b64:
                return jsonify({"error": "Falta la foto en base64"}), 400

            # Decodifica y procesa la imagen
            try:
                img_bytes = base64.b64decode(foto_b64)
            except Exception:
                return jsonify({"error": "Base64 inválido"}), 400

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            try:
                tmp.write(img_bytes)
                tmp.flush()
                firma_nueva = sign_image(tmp.name, self.priv_key)
                
                # Guardar firma en BD
                insertar_firma_bd(firma_nueva)
                
            finally:
                tmp.close()
                os.unlink(tmp.name)

            # Obtener y comparar firmas de BD
            firmas_db = self.obtener_firmas_db()
            if not firmas_db:
                return jsonify({"mensaje": "Primera firma guardada correctamente"}), 200

            # Comparar con firmas existentes
            max_sim = -1.0
            firma_mas_similar = None
            for firma_db in firmas_db[:-1]:  # Excluir la última (recién insertada)
                try:
                    sim = self.comparator.compare(firma_nueva, firma_db)
                    if sim > max_sim:
                        max_sim = sim
                        firma_mas_similar = firma_db
                except Exception:
                    continue

            if firma_mas_similar is None:
                return jsonify({"mensaje": "Firma guardada, sin coincidencias"}), 200

            return jsonify({
                "firma_mas_similar": firma_mas_similar,
                "similitud": round(max_sim * 100, 2),
                "mensaje": "Firma procesada y comparada"
            })
        
        @self.app.route('/upload', methods=['POST'])
        def upload_image():
            """Endpoint para recibir imágenes y procesarlas"""
            if 'image' not in request.files:
                return jsonify({"error": "No se encontró imagen"}), 400
            
            file = request.files['image']
            if file.filename == '':
                return jsonify({"error": "No se seleccionó archivo"}), 400
            
            # Guardar imagen
            filename = f"recibida_{int(time.time())}.jpg"
            filepath = os.path.join(self.carpeta_capturas, filename)
            file.save(filepath)
            
            # Procesar imagen (generar firma y guardar en BD)
            try:
                firma = sign_image(filepath, self.priv_key)
                insertar_firma_bd(firma)
                return jsonify({"mensaje": "Imagen procesada y firma guardada", "archivo": filename}), 200
            except Exception as e:
                return jsonify({"error": f"Error procesando imagen: {str(e)}"}), 500
    
    def obtener_firmas_db(self):
        """Obtiene todas las firmas de la base de datos"""
        try:
            with psycopg2.connect(**db_params) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT firma FROM firmas")
                    return [row[0] for row in cur.fetchall()]
        except Exception as e:
            print(f"Error obteniendo firmas de BD: {e}")
            return []
    
    def procesar_frame_con_caras(self, frame):
        """Procesa un frame detectando caras y generando firmas"""
        if frame is None:
            return
            
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        caras = self.detector_caras.detectMultiScale(gris, 1.1, 4)
        
        if len(caras) > 0:
            for i, (x, y, w, h) in enumerate(caras):
                # Extraer cara
                cara_recortada = frame[y:y+h, x:x+w]
                
                # Guardar imagen
                nombre_archivo = f"{self.carpeta_capturas}/cara_{self.contador:05}_{i}.jpg"
                cv2.imwrite(nombre_archivo, cara_recortada)
                
                # Generar y guardar firma
                try:
                    firma = sign_image(nombre_archivo, self.priv_key)
                    insertar_firma_bd(firma)
                    print(f"✓ Cara procesada y firma guardada: {nombre_archivo}")
                except Exception as e:
                    print(f"✗ Error procesando {nombre_archivo}: {e}")
            
            self.contador += 1
    
    def capturar_desde_ip_camera(self, ip_camera, username=None, password=None):
        """Captura desde cámara IP con procesamiento integrado"""
        stream_urls = [
            f"http://{ip_camera}/video/mjpg.cgi",
            f"http://{ip_camera}/mjpg/video.mjpg",
            f"http://{ip_camera}/videostream.cgi",
            f"http://{ip_camera}:8080/video",
            f"rtsp://{ip_camera}/stream1"
        ]
        
        for url in stream_urls:
            try:
                print(f"🔍 Intentando conectar a: {url}")
                
                if username and password:
                    url = url.replace("://", f"://{username}:{password}@")
                
                cap = cv2.VideoCapture(url)
                
                if cap.isOpened():
                    print(f"✅ Conectado exitosamente a {url}")
                    self.running = True
                    
                    while self.running:
                        ret, frame = cap.read()
                        
                        if ret:
                            # Procesar frame
                            self.procesar_frame_con_caras(frame)
                            
                            # Mostrar frame (opcional)
                            cv2.imshow('IP Camera Feed', frame)
                            
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                self.running = False
                                break
                                
                        else:
                            print("⚠️ Error al leer frame")
                            break
                        
                        time.sleep(0.1)
                    
                    cap.release()
                    cv2.destroyAllWindows()
                    return True
                    
            except Exception as e:
                print(f"❌ Error con {url}: {e}")
                continue
        
        print("❌ No se pudo conectar a ningún stream de la cámara")
        return False
    
    def iniciar_monitoreo_y_envio(self):
        """Inicia monitoreo de carpeta y envío automático"""
        def enviar_imagenes():
            print(f"📤 Iniciando envío automático a {self.ip_destino}:{self.puerto}")
            monitorear_y_enviar(
                carpeta_capturas=self.carpeta_capturas,
                ip_destino=self.ip_destino,
                puerto=self.puerto,
                intervalo=5
            )
        
        thread_envio = threading.Thread(target=enviar_imagenes, daemon=True)
        thread_envio.start()
        return thread_envio
    
    def ejecutar_servidor(self, host="0.0.0.0", port=5000):
        """Ejecuta el servidor Flask"""
        print(f"🔑 Iniciando servidor en {host}:{port}")
        self.app.run(host=host, port=port, debug=False)
    
    def ejecutar_captura_completa(self, ip_camera, username=None, password=None):
        """Ejecuta captura completa con procesamiento de firmas"""
        print("=== Iniciando sistema completo ===")
        print(f"📷 Cámara IP: {ip_camera}")
        print(f"📁 Carpeta capturas: {self.carpeta_capturas}")
        print(f"🌐 Servidor: {self.ip_destino}:{self.puerto}")
        
        # Iniciar envío automático si es necesario
        # thread_envio = self.iniciar_monitoreo_y_envio()
        
        try:
            # Capturar desde cámara IP
            self.capturar_desde_ip_camera(ip_camera, username, password)
        except KeyboardInterrupt:
            print("\n⏹️ Deteniendo captura...")
            self.running = False
        
        print("✅ Flujo completado")

# Funciones principales del programa

def run_server(host="0.0.0.0", port=5000):
    """Arranca el servidor REST para firma y comparación."""
    sistema = CapturaYFirma(puerto=port)
    sistema.ejecutar_servidor(host, port)

def run_camera_flow(ip_camera, dest_ip, dest_port, interval=5):
    """Ejecuta sólo la captura+envío de imágenes con procesamiento de firmas."""
    print(f"📷 Iniciando CAMERA flow contra {dest_ip}:{dest_port}")
    sistema = CapturaYFirma(
        carpeta_capturas="capturas",
        ip_destino=dest_ip,
        puerto=dest_port
    )
    sistema.ejecutar_captura_completa(ip_camera)

def run_full(ip_camera, dest_ip, dest_port, host="0.0.0.0", port=5000):
    """Ejecuta both: server REST + captura/envío con procesamiento completo."""
    sistema = CapturaYFirma(
        carpeta_capturas="capturas",
        ip_destino=dest_ip,
        puerto=dest_port
    )
    
    # Ejecutar servidor en hilo separado
    def servidor_thread():
        sistema.ejecutar_servidor(host, port)
    
    t = threading.Thread(target=servidor_thread, daemon=True)
    t.start()
    
    # Dar tiempo al servidor para iniciar
    time.sleep(2)
    
    # Ejecutar captura en el hilo principal
    sistema.ejecutar_captura_completa(ip_camera)

def capturar_foto_y_procesar():
    """Captura una foto desde webcam y la procesa"""
    sistema = CapturaYFirma()
    
    try:
        # Capturar foto cuadrada
        filename = capture_square_photo('captura_manual.jpg')
        print(f"📸 Foto capturada: {filename}")
        
        # Generar firma
        firma = sign_image(filename, sistema.priv_key)
        
        # Guardar en BD
        insertar_firma_bd(firma)
        print("✅ Firma generada y guardada en base de datos")
        
        return filename, firma
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

def escanear_y_conectar():
    """Escanea la red en busca de cámaras IP"""
    print("🔍 Escaneando red en busca de cámaras...")
    camaras = scan_network_for_cameras()
    
    if camaras:
        print(f"📷 Cámaras encontradas: {camaras}")
        return camaras
    else:
        print("❌ No se encontraron cámaras en la red")
        return []

def procesar_imagenes_existentes(carpeta="capturas"):
    """Procesa todas las imágenes existentes en una carpeta"""
    sistema = CapturaYFirma(carpeta_capturas=carpeta)
    
    if not os.path.exists(carpeta):
        print(f"❌ La carpeta {carpeta} no existe")
        return
    
    extensiones = ('.jpg', '.jpeg', '.png', '.bmp')
    imagenes = [f for f in os.listdir(carpeta) if f.lower().endswith(extensiones)]
    
    if not imagenes:
        print(f"❌ No se encontraron imágenes en {carpeta}")
        return
    
    print(f"📁 Procesando {len(imagenes)} imágenes...")
    
    for imagen in imagenes:
        filepath = os.path.join(carpeta, imagen)
        try:
            # Generar firma
            firma = sign_image(filepath, sistema.priv_key)
            
            # Guardar en BD
            insertar_firma_bd(firma)
            print(f"✅ Procesada: {imagen}")
            
        except Exception as e:
            print(f"❌ Error procesando {imagen}: {e}")
    
    print("✅ Procesamiento completado")

def main():
    p = argparse.ArgumentParser(description="Sistema Integrado de Captura y Procesamiento de Firmas")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Servidor
    srv = sub.add_parser("server", help="Arranca API REST de firma/comparación")
    srv.add_argument("--host", default="0.0.0.0", help="Host del servidor")
    srv.add_argument("--port", type=int, default=5000, help="Puerto del servidor")

    # Cámara
    cam = sub.add_parser("camera", help="Captura desde cámara IP y procesa firmas")
    cam.add_argument("ip_camera", help="IP de la cámara (o 0 para webcam)")
    cam.add_argument("--dest-ip", default="192.168.1.100", help="IP destino para envío")
    cam.add_argument("--dest-port", type=int, default=8080, help="Puerto destino")
    cam.add_argument("--username", help="Usuario para cámara IP")
    cam.add_argument("--password", help="Contraseña para cámara IP")

    # Todo junto
    full = sub.add_parser("full", help="Servidor + captura en paralelo")
    full.add_argument("ip_camera", help="IP de la cámara (o 0 para webcam)")
    full.add_argument("--dest-ip", default="192.168.1.100", help="IP destino")
    full.add_argument("--dest-port", type=int, default=8080, help="Puerto destino")
    full.add_argument("--host", default="0.0.0.0", help="Host del servidor")
    full.add_argument("--port", type=int, default=5000, help="Puerto del servidor")
    full.add_argument("--username", help="Usuario para cámara IP")
    full.add_argument("--password", help="Contraseña para cámara IP")

    # Foto manual
    foto = sub.add_parser("foto", help="Captura foto manual desde webcam")

    # Escanear red
    scan = sub.add_parser("scan", help="Escanea red en busca de cámaras IP")

    # Procesar imágenes existentes
    proc = sub.add_parser("process", help="Procesa imágenes existentes en carpeta")
    proc.add_argument("--carpeta", default="capturas", help="Carpeta con imágenes")

    # Enviar imágenes
    send = sub.add_parser("send", help="Envía imágenes a servidor")
    send.add_argument("--carpeta", default="capturas", help="Carpeta con imágenes")
    send.add_argument("--dest-ip", default="192.168.1.100", help="IP destino")
    send.add_argument("--dest-port", type=int, default=5000, help="Puerto destino")
    send.add_argument("--endpoint", default="/comparar", help="Endpoint destino")

    # Probar base de datos
    testdb = sub.add_parser("testdb", help="Prueba la conexión a la base de datos")

    args = p.parse_args()

    try:
        if args.cmd == "server":
            run_server(args.host, args.port)
        
        elif args.cmd == "camera":
            sistema = CapturaYFirma(
                carpeta_capturas="capturas",
                ip_destino=args.dest_ip,
                puerto=args.dest_port
            )
            sistema.ejecutar_captura_completa(
                args.ip_camera, 
                username=getattr(args, 'username', None),
                password=getattr(args, 'password', None)
            )
        
        elif args.cmd == "full":
            run_full(
                args.ip_camera, 
                args.dest_ip, 
                args.dest_port, 
                args.host, 
                args.port
            )
        
        elif args.cmd == "foto":
            filename, firma = capturar_foto_y_procesar()
            if filename:
                print(f"📄 Archivo: {filename}")
                print(f"🔏 Firma: {firma[:50]}...")
        
        elif args.cmd == "scan":
            camaras = escanear_y_conectar()
            if camaras:
                print("\n📋 Cámaras disponibles:")
                for ip in camaras:
                    print(f"  • {ip}")
        
        elif args.cmd == "process":
            procesar_imagenes_existentes(args.carpeta)
        
        elif args.cmd == "send":
            print(f"📤 Enviando imágenes de {args.carpeta} a {args.dest_ip}:{args.dest_port}")
            enviar_imagenes_a_ip(
                carpeta_capturas=args.carpeta,
                ip_destino=args.dest_ip,
                puerto=args.dest_port,
                endpoint=args.endpoint
            )
        
        elif args.cmd == "testdb":
            print("🔍 Probando conexión a la base de datos...")
            print(f"📋 DATABASE_URL: {os.getenv('DATABASE_URL', 'No configurada')}")
            if probar_conexion_bd():
                crear_tabla_firmas()
                print("✅ Base de datos lista para usar")
            else:
                print("❌ Problemas con la base de datos")
                print("\n💡 Sugerencias:")
                print("  1. Verifica que PostgreSQL esté ejecutándose")
                print("  2. Confirma las credenciales en el archivo .env")
                print("  3. Asegúrate de que la base de datos existe")
            
    except KeyboardInterrupt:
        print("\n⏹️ Programa interrumpido por el usuario")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
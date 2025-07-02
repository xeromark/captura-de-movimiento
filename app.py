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
DB_PARAMS = get_db_params()

def crear_tabla_firmas():
    """Crea la tabla de firmas si no existe"""
    try:
        with psycopg2.connect(**DB_PARAMS) as conn:
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

def probar_conexion_bd():
    """Prueba la conexión a la base de datos"""
    try:
        with psycopg2.connect(**DB_PARAMS) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()[0]
                print(f"✅ Conexión BD exitosa: {version}")
                return True
    except Exception as e:
        print(f"❌ Error conectando a BD: {e}")
        print(f"📋 Parámetros BD: {DB_PARAMS}")
        return False

class IntegratedSystem:
    def __init__(self, model_path=MODEL_PATH, device=DEVICE, face_threshold=0.7):
        # Inicializar componentes
        self.model_path = model_path
        self.device = device
        self.face_threshold = face_threshold  # Threshold para detección de rostros
        self.running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.carpeta_capturas = os.path.join(ROOT, "capturas")
        os.makedirs(self.carpeta_capturas, exist_ok=True)
        
        # Verificar conexión BD y crear tabla si es necesario
        print(f"📋 Conectando a BD: {DB_PARAMS['host']}:{DB_PARAMS['port']}")
        if probar_conexion_bd():
            crear_tabla_firmas()
        else:
            print("⚠️ Continuando sin conexión a BD")
        
        # Cargar el comparador
        try:
            self.comparator = SignatureComparator(self.model_path, device=self.device)
            self.priv_key, self.pub_key = generate_keys()
            print(f"✅ Modelo cargado correctamente desde {self.model_path}")
        except Exception as e:
            print(f"❌ Error al cargar el modelo: {e}")
            self.comparator = None

        # Cargar detector de caras
        self.detector_caras = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print(f"🎯 Threshold de detección facial: {self.face_threshold}")
        
        # Cargar firmas de la base de datos
        self.firmas_db = self.obtener_firmas_db()
        print(f"📊 Se cargaron {len(self.firmas_db)} firmas de la base de datos")
    
    def obtener_firmas_db(self):
        """Lee todas las firmas de la base de datos."""
        try:
            with psycopg2.connect(**DB_PARAMS) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT firma FROM firmas")
                    return [row[0] for row in cur.fetchall()]
        except Exception as e:
            print(f"Error al conectar a la base de datos: {e}")
            return []
    
    def insertar_firma(self, firma):
        """Inserta una nueva firma en la base de datos."""
        try:
            with psycopg2.connect(**DB_PARAMS) as conn:
                with conn.cursor() as cur:
                    cur.execute("INSERT INTO firmas (firma) VALUES (%s)", (firma,))
                conn.commit()
            print("✓ Firma guardada en la base de datos")
            # Actualizar lista de firmas
            self.firmas_db = self.obtener_firmas_db()
            return True
        except Exception as e:
            print(f"Error al guardar firma: {e}")
            return False
    
    def comparar_firma_con_db(self, firma_nueva):
        """Compara una firma con todas las almacenadas en la DB."""
        if not self.firmas_db:
            return None, 0.0
        
        max_sim = -1.0
        firma_mas_similar = None
        
        for firma_db in self.firmas_db:
            try:
                sim = self.comparator.compare(firma_nueva, firma_db)
                if sim > max_sim:
                    max_sim = sim
                    firma_mas_similar = firma_db
            except Exception as e:
                continue
                
        return firma_mas_similar, max_sim
    
    def detectar_caras_con_threshold(self, frame_gris):
        """Detecta caras aplicando threshold de confianza"""
        # Detectar caras con parámetros ajustables
        caras = self.detector_caras.detectMultiScale(
            frame_gris, 
            scaleFactor=1.1,           # Factor de escala entre niveles de imagen
            minNeighbors=4,            # Mínimo número de vecinos para considerar detección válida
            minSize=(30, 30),          # Tamaño mínimo de cara
            maxSize=(300, 300)         # Tamaño máximo de cara
        )
        
        # Filtrar caras por tamaño (threshold adicional)
        caras_filtradas = []
        for (x, y, w, h) in caras:
            # Solo considerar caras que cumplan con el threshold de tamaño
            area = w * h
            if area > (50 * 50):  # Mínimo 50x50 píxeles
                caras_filtradas.append((x, y, w, h))
        
        return caras_filtradas
    
    def procesar_cara(self, frame, x, y, w, h):
        """Procesa una cara detectada en el frame."""
        # Recortar la cara
        cara = frame[y:y+h, x:x+w]
        
        # Verificar que la cara tenga un tamaño mínimo
        if cara.shape[0] < 30 or cara.shape[1] < 30:
            return None, 0.0
        
        # Guardar temporalmente
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        try:
            cv2.imwrite(tmp.name, cara)
            # Generar firma
            firma = sign_image(tmp.name, self.priv_key)
            # Comparar con firmas existentes
            _, similitud = self.comparar_firma_con_db(firma)
            
            # Aplicar threshold de similitud para considerarlo una coincidencia válida
            if similitud > self.face_threshold:
                return firma, similitud * 100  # Convertir a porcentaje
            else:
                return firma, similitud * 100  # Devolver de cualquier manera para mostrar
                
        except Exception as e:
            print(f"Error al procesar cara: {e}")
            return None, 0.0
        finally:
            tmp.close()
            try:
                os.unlink(tmp.name)
            except:
                pass
    
    def es_coincidencia_valida(self, similitud):
        """Determina si una similitud es considerada una coincidencia válida"""
        return similitud >= self.face_threshold
    
    def iniciar_captura_webcam(self):
        """Inicia la captura desde la webcam local."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("No se pudo acceder a la webcam local.")
            return False
        
        print(f"Webcam conectada. Threshold: {self.face_threshold}")
        print("Presiona 'espacio' para guardar firma, 'q' para salir.")
        self.running = True
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Error al leer frame de la webcam.")
                break
            
            # Convertir a escala de grises para detección
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detectar caras con threshold
            caras = self.detectar_caras_con_threshold(gris)
            
            # Procesar cada cara
            for (x, y, w, h) in caras:
                # Procesar cara solo si tenemos comparador
                if self.comparator:
                    firma, similitud = self.procesar_cara(frame, x, y, w, h)
                    
                    # Determinar color del rectángulo según threshold
                    if self.es_coincidencia_valida(similitud / 100):
                        color = (0, 255, 0)  # Verde para coincidencias válidas
                        texto = f"MATCH: {similitud:.2f}%"
                    else:
                        color = (0, 165, 255)  # Naranja para similitudes bajas
                        texto = f"{similitud:.2f}%"
                    
                    # Dibujar rectángulo con color según threshold
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Mostrar porcentaje de similitud
                    cv2.putText(frame, texto, (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                color, 2)
                    
                    # Guardar firma al presionar espacio
                    key = cv2.waitKey(1) & 0xFF
                    if key == 32:  # Tecla espacio
                        if firma:
                            if self.insertar_firma(firma):
                                print(f"Firma guardada para cara en ({x},{y}) - Similitud: {similitud:.2f}%")
                                # Guardar imagen de la cara
                                img_path = os.path.join(self.carpeta_capturas, f"cara_{time.time():.0f}.jpg")
                                cv2.imwrite(img_path, frame[y:y+h, x:x+w])
                                print(f"Imagen guardada en {img_path}")
                else:
                    # Si no hay comparador, solo dibujar rectángulo básico
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Mostrar información del threshold en pantalla
            info_text = f"Threshold: {self.face_threshold} | Caras: {len(caras)}"
            cv2.putText(frame, info_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Mostrar frame
            cv2.imshow("Captura de Caras", frame)
            
            # Comprobar tecla 'q' para salir
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
                break
        
        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()
        return True
    
    def iniciar_captura_ip_camera(self, ip_address, username=None, password=None):
        """Inicia la captura desde una cámara IP."""
        # URLs comunes para cámaras IP
        stream_urls = [
            f"http://{ip_address}/video/mjpg.cgi",
            f"http://{ip_address}/mjpg/video.mjpg",
            f"http://{ip_address}/videostream.cgi",
            f"http://{ip_address}:8080/video",
            f"rtsp://{ip_address}/stream1"
        ]
        
        for url in stream_urls:
            try:
                print(f"Intentando conectar a: {url}")
                
                # Agregar autenticación si se proporciona
                if username and password:
                    parsed_url = urlparse(url)
                    url = f"{parsed_url.scheme}://{username}:{password}@{parsed_url.netloc}{parsed_url.path}"
                
                cap = cv2.VideoCapture(url)
                
                if cap.isOpened():
                    print(f"Conectado exitosamente a {url}")
                    print("Presiona 'espacio' para guardar firma, 'q' para salir.")
                    self.running = True
                    
                    while self.running:
                        ret, frame = cap.read()
                        
                        if not ret:
                            print("Error al leer frame de la cámara IP.")
                            break
                        
                        # Convertir a escala de grises para detección
                        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        # Detectar caras
                        caras = self.detector_caras.detectMultiScale(gris, 1.1, 4)
                        
                        # Procesar cada cara
                        for (x, y, w, h) in caras:
                            # Dibujar rectángulo
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            
                            # Procesar cara solo si tenemos comparador
                            if self.comparator:
                                firma, similitud = self.procesar_cara(frame, x, y, w, h)
                                # Mostrar porcentaje de similitud
                                texto = f"{similitud:.2f}%"
                                cv2.putText(frame, texto, (x, y-10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                                            (0, 255, 0), 2)
                        
                        # Mostrar frame
                        cv2.imshow("Captura de Caras (IP)", frame)
                        
                        # Comprobar tecla 'q' para salir y 'espacio' para guardar
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            self.running = False
                            break
                        elif key == 32:  # Tecla espacio
                            # Guardar la última cara procesada
                            if caras.size > 0:
                                x, y, w, h = caras[0]
                                firma, _ = self.procesar_cara(frame, x, y, w, h)
                                if firma:
                                    if self.insertar_firma(firma):
                                        print(f"Firma guardada para cara en ({x},{y})")
                                        # Guardar imagen de la cara
                                        img_path = os.path.join(self.carpeta_capturas, f"cara_{time.time()}.jpg")
                                        cv2.imwrite(img_path, frame[y:y+h, x:x+w])
                                        print(f"Imagen guardada en {img_path}")
                    
                    cap.release()
                    cv2.destroyAllWindows()
                    return True
                    
            except Exception as e:
                print(f"Error con {url}: {e}")
                continue
        
        print("No se pudo conectar a ningún stream de la cámara IP.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Sistema Integrado de Captura de Movimiento y Procesamiento de Firmas")
    
    # Crear subparsers para diferentes modos
    subparsers = parser.add_subparsers(dest='mode', help='Modos de operación disponibles')
    
    # Modo cámara (por defecto)
    camera_parser = subparsers.add_parser('camera', help='Modo cámara con detección facial')
    camera_parser.add_argument("--ip", help="Dirección IP de la cámara (opcional)")
    camera_parser.add_argument("--username", help="Usuario para la cámara IP (opcional)")
    camera_parser.add_argument("--password", help="Contraseña para la cámara IP (opcional)")
    camera_parser.add_argument("--face-threshold", type=float, default=0.7, 
                              help="Threshold para detección de rostros (0.0-1.0, default: 0.7)")
    
    # Modo servidor API
    server_parser = subparsers.add_parser('server', help='Iniciar servidor REST API')
    server_parser.add_argument("--port", type=int, default=5000, help="Puerto del servidor (default: 5000)")
    server_parser.add_argument("--face-threshold", type=float, default=0.7, 
                              help="Threshold para detección de rostros (0.0-1.0, default: 0.7)")
    
    # Modo foto (captura única)
    photo_parser = subparsers.add_parser('foto', help='Captura una foto con detección facial')
    photo_parser.add_argument("--output", "-o", help="Archivo de salida (opcional)")
    photo_parser.add_argument("--face-threshold", type=float, default=0.7, 
                             help="Threshold para detección de rostros (0.0-1.0, default: 0.7)")
    
    # Modo procesamiento
    process_parser = subparsers.add_parser('process', help='Procesar firma desde archivo')
    process_parser.add_argument("input_file", help="Archivo de imagen a procesar")
    process_parser.add_argument("--face-threshold", type=float, default=0.7, 
                               help="Threshold para detección de rostros (0.0-1.0, default: 0.7)")
    
    # Modo test BD
    test_parser = subparsers.add_parser('testdb', help='Probar conexión a base de datos')
    
    # Modo full (completo)
    full_parser = subparsers.add_parser('full', help='Sistema completo con todas las funciones')
    full_parser.add_argument("--ip", help="Dirección IP de la cámara (opcional)")
    full_parser.add_argument("--username", help="Usuario para la cámara IP (opcional)")
    full_parser.add_argument("--password", help="Contraseña para la cámara IP (opcional)")
    full_parser.add_argument("--face-threshold", type=float, default=0.7, 
                            help="Threshold para detección de rostros (0.0-1.0, default: 0.7)")
    full_parser.add_argument("--port", type=int, default=5000, help="Puerto del servidor (default: 5000)")
    
    # Para mantener compatibilidad con argumentos antiguos
    parser.add_argument("--ip", help="Dirección IP de la cámara (opcional)")
    parser.add_argument("--username", help="Usuario para la cámara IP (opcional)")
    parser.add_argument("--password", help="Contraseña para la cámara IP (opcional)")
    parser.add_argument("--face-threshold", type=float, default=0.7, 
                       help="Threshold para detección de rostros (0.0-1.0, default: 0.7)")
    parser.add_argument("--testdb", action="store_true", help="Solo probar conexión a BD")
    
    args = parser.parse_args()
    
    # Compatibilidad con modo anterior
    if args.testdb:
        args.mode = 'testdb'
    elif args.mode is None:
        args.mode = 'camera'  # Modo por defecto
    
    # Manejar modo test BD
    if args.mode == 'testdb':
        print("🔍 Probando conexión a la base de datos...")
        print(f"📋 DATABASE_URL: {os.getenv('DATABASE_URL', 'No configurada')}")
        print(f"📋 Parámetros: {DB_PARAMS}")
        if probar_conexion_bd():
            crear_tabla_firmas()
            print("✅ Base de datos lista para usar")
        else:
            print("❌ Problemas con la base de datos")
        return
    
    # Validar threshold
    if not 0.0 <= args.face_threshold <= 1.0:
        print("❌ Error: --face-threshold debe estar entre 0.0 y 1.0")
        return
    
    # Inicializar sistema con threshold configurado
    sistema = IntegratedSystem(face_threshold=args.face_threshold)
    
    print(f"🎯 Modo seleccionado: {args.mode}")
    print(f"🔍 Threshold de detección facial: {args.face_threshold}")
    
    # Ejecutar según el modo seleccionado
    if args.mode == 'server':
        print(f"🌐 Iniciando servidor en puerto {args.port}")
        sistema.iniciar_servidor_api(puerto=args.port)
        
    elif args.mode == 'foto':
        print("📸 Modo captura de foto")
        sistema.capturar_foto_con_rostro(args.output)
        
    elif args.mode == 'process':
        print(f"⚙️ Procesando archivo: {args.input_file}")
        sistema.procesar_imagen_archivo(args.input_file)
        
    elif args.mode == 'full':
        print("🔄 Modo completo: servidor + cámara")
        # Iniciar servidor en segundo plano
        import threading
        servidor_thread = threading.Thread(target=sistema.iniciar_servidor_api, 
                                         kwargs={'puerto': args.port})
        servidor_thread.daemon = True
        servidor_thread.start()
        print(f"🌐 Servidor API iniciado en puerto {args.port}")
        
        # Iniciar cámara
        iniciar_camara_con_args(sistema, args)
        
    elif args.mode == 'camera':
        print("📹 Modo cámara con detección facial")
        iniciar_camara_con_args(sistema, args)

def iniciar_camara_con_args(sistema, args):
    """Función auxiliar para iniciar cámara con argumentos"""
    
    # Intentar conectar a la cámara
    if args.ip:
        # Usar cámara IP especificada
        sistema.iniciar_captura_ip_camera(args.ip, args.username, args.password)
    else:
        # Intentar usar webcam local
        if not sistema.iniciar_captura_webcam():
            # Si falla, preguntar por cámara IP
            print("\nNo se pudo acceder a la webcam local.")
            print("¿Deseas intentar con una cámara IP? (s/n)")
            respuesta = input().lower()
            
            if respuesta == 's':
                print("Ingresa la dirección IP de la cámara:")
                ip = input()
                print("Usuario (opcional):")
                username = input() or None
                print("Contraseña (opcional):")
                password = input() or None
                
                sistema.iniciar_captura_ip_camera(ip, username, password)
            else:
                print("Finalizando programa.")

if __name__ == "__main__":
    main()
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
# Configuración de base de datos desde variables de entorno
DB_PARAMS = {
    'dbname': os.getenv('DB_NAME', 'tu_basededatos'),
    'user': os.getenv('DB_USER', 'tu_usuario'),
    'password': os.getenv('DB_PASSWORD', 'tu_contraseña'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432))
}

class IntegratedSystem:
    def __init__(self, model_path=MODEL_PATH, device=DEVICE):
        # Inicializar componentes
        self.model_path = model_path
        self.device = device
        self.running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.carpeta_capturas = os.path.join(ROOT, "capturas")
        os.makedirs(self.carpeta_capturas, exist_ok=True)
        
        # Cargar el comparador
        try:
            self.comparator = SignatureComparator(self.model_path, device=self.device)
            self.priv_key, self.pub_key = generate_keys()
            print(f"Modelo cargado correctamente desde {self.model_path}")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            self.comparator = None

        # Cargar detector de caras
        self.detector_caras = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Cargar firmas de la base de datos
        self.firmas_db = self.obtener_firmas_db()
        print(f"Se cargaron {len(self.firmas_db)} firmas de la base de datos")
    
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
    
    def procesar_cara(self, frame, x, y, w, h):
        """Procesa una cara detectada en el frame."""
        # Recortar la cara
        cara = frame[y:y+h, x:x+w]
        
        # Guardar temporalmente
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        try:
            cv2.imwrite(tmp.name, cara)
            # Generar firma
            firma = sign_image(tmp.name, self.priv_key)
            # Comparar con firmas existentes
            _, similitud = self.comparar_firma_con_db(firma)
            return firma, similitud * 100  # Convertir a porcentaje
        except Exception as e:
            print(f"Error al procesar cara: {e}")
            return None, 0.0
        finally:
            tmp.close()
            try:
                os.unlink(tmp.name)
            except:
                pass
    
    def iniciar_captura_webcam(self):
        """Inicia la captura desde la webcam local."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("No se pudo acceder a la webcam local.")
            return False
        
        print("Webcam conectada. Presiona 'espacio' para guardar firma, 'q' para salir.")
        self.running = True
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Error al leer frame de la webcam.")
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
                    
                    # Guardar firma al presionar espacio
                    key = cv2.waitKey(1) & 0xFF
                    if key == 32:  # Tecla espacio
                        if firma:
                            if self.insertar_firma(firma):
                                print(f"Firma guardada para cara en ({x},{y})")
                                # Guardar imagen de la cara
                                img_path = os.path.join(self.carpeta_capturas, f"cara_{time.time()}.jpg")
                                cv2.imwrite(img_path, frame[y:y+h, x:x+w])
                                print(f"Imagen guardada en {img_path}")
            
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
    parser.add_argument("--ip", help="Dirección IP de la cámara (opcional)")
    parser.add_argument("--username", help="Usuario para la cámara IP (opcional)")
    parser.add_argument("--password", help="Contraseña para la cámara IP (opcional)")
    args = parser.parse_args()
    
    # Inicializar sistema
    sistema = IntegratedSystem()
    
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
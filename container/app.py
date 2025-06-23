import threading
import time
import cv2
import os
from collections import deque
from external_cam import capture_from_ip_camera, scan_network_for_cameras
from camera_photo import capturar_movimiento
from sender import monitorear_y_enviar
import queue
import socket
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BufferItem:
    """Clase para representar un elemento en el buffer"""
    frame: np.ndarray
    timestamp: float
    metadata: dict
    processed: bool = False

@dataclass
class DetectedFace:
    """Clase para representar una cara detectada"""
    image: np.ndarray
    coordinates: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    timestamp: float
    metadata: dict

class CircularBuffer:
    """Buffer circular thread-safe para manejo eficiente de frames"""
    
    def __init__(self, maxsize: int = 100):
        self.maxsize = maxsize
        self.buffer = deque(maxlen=maxsize)
        self.lock = threading.RLock()
        self.not_empty = threading.Condition(self.lock)
        self.stats = {
            'items_added': 0,
            'items_consumed': 0,
            'buffer_overflows': 0
        }
    
    def put(self, item: BufferItem, block: bool = True, timeout: Optional[float] = None):
        """Añade un elemento al buffer"""
        with self.lock:
            if len(self.buffer) >= self.maxsize:
                self.stats['buffer_overflows'] += 1
                if not block:
                    return False
                
            self.buffer.append(item)
            self.stats['items_added'] += 1
            self.not_empty.notify()
            return True
    
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[BufferItem]:
        """Obtiene un elemento del buffer"""
        with self.not_empty:
            if not block and len(self.buffer) == 0:
                return None
                
            if timeout:
                self.not_empty.wait_for(lambda: len(self.buffer) > 0, timeout=timeout)
            else:
                self.not_empty.wait_for(lambda: len(self.buffer) > 0)
            
            if len(self.buffer) > 0:
                item = self.buffer.popleft()
                self.stats['items_consumed'] += 1
                return item
            return None
    
    def size(self) -> int:
        """Retorna el tamaño actual del buffer"""
        with self.lock:
            return len(self.buffer)
    
    def is_full(self) -> bool:
        """Verifica si el buffer está lleno"""
        with self.lock:
            return len(self.buffer) >= self.maxsize
    
    def clear(self):
        """Limpia el buffer"""
        with self.lock:
            self.buffer.clear()
    
    def get_stats(self) -> dict:
        """Retorna estadísticas del buffer"""
        with self.lock:
            return {
                **self.stats,
                'current_size': len(self.buffer),
                'max_size': self.maxsize,
                'usage_percentage': (len(self.buffer) / self.maxsize) * 100
            }

class ProcessingPipeline:
    """Pipeline de procesamiento con múltiples buffers"""
    
    def __init__(self, buffer_size: int = 100):
        # Buffers para diferentes etapas
        self.raw_frame_buffer = CircularBuffer(buffer_size)
        self.processed_frame_buffer = CircularBuffer(buffer_size)
        self.face_detection_buffer = CircularBuffer(buffer_size // 2)
        self.send_queue = queue.PriorityQueue(maxsize=buffer_size // 4)
        
        # Control de hilos
        self.running = False
        self.threads = []
        
        # Detector de caras
        self.detector_caras = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Contadores y estadísticas
        self.frame_counter = 0
        self.face_counter = 0
        self.stats = {
            'frames_processed': 0,
            'faces_detected': 0,
            'processing_errors': 0
        }
    
    def add_frame(self, frame: np.ndarray, metadata: dict = None) -> bool:
        """Añade un frame al pipeline"""
        if not self.running:
            return False
        
        buffer_item = BufferItem(
            frame=frame.copy(),
            timestamp=time.time(),
            metadata=metadata or {},
            processed=False
        )
        
        success = self.raw_frame_buffer.put(buffer_item, block=False)
        if success:
            self.frame_counter += 1
        
        return success
    
    def process_frames_worker(self):
        """Worker para procesar frames del buffer raw"""
        logger.info("Iniciando worker de procesamiento de frames")
        
        while self.running:
            try:
                buffer_item = self.raw_frame_buffer.get(timeout=1.0)
                if buffer_item is None:
                    continue
                
                # Procesar frame (convertir a escala de grises, etc.)
                gray_frame = cv2.cvtColor(buffer_item.frame, cv2.COLOR_BGR2GRAY)
                
                # Actualizar metadata
                buffer_item.metadata.update({
                    'processed_timestamp': time.time(),
                    'frame_size': buffer_item.frame.shape,
                    'processing_time': time.time() - buffer_item.timestamp
                })
                
                buffer_item.processed = True
                
                # Mover al buffer de frames procesados
                self.processed_frame_buffer.put(buffer_item, block=False)
                self.stats['frames_processed'] += 1
                
            except Exception as e:
                logger.error(f"Error en procesamiento de frames: {e}")
                self.stats['processing_errors'] += 1
    
    def face_detection_worker(self):
        """Worker para detección de caras"""
        logger.info("Iniciando worker de detección de caras")
        
        while self.running:
            try:
                buffer_item = self.processed_frame_buffer.get(timeout=1.0)
                if buffer_item is None:
                    continue
                
                # Detectar caras
                gray = cv2.cvtColor(buffer_item.frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector_caras.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    for i, (x, y, w, h) in enumerate(faces):
                        # Extraer cara
                        face_img = buffer_item.frame[y:y+h, x:x+w]
                        
                        detected_face = DetectedFace(
                            image=face_img,
                            coordinates=(x, y, w, h),
                            confidence=1.0,  # Podríamos calcular confianza real
                            timestamp=buffer_item.timestamp,
                            metadata={
                                **buffer_item.metadata,
                                'face_index': i,
                                'face_id': f"{self.face_counter:05d}_{i}",
                                'detection_time': time.time()
                            }
                        )
                        
                        # Añadir al buffer de caras detectadas
                        face_buffer_item = BufferItem(
                            frame=face_img,
                            timestamp=buffer_item.timestamp,
                            metadata=detected_face.metadata,
                            processed=True
                        )
                        
                        self.face_detection_buffer.put(face_buffer_item, block=False)
                    
                    self.face_counter += 1
                    self.stats['faces_detected'] += len(faces)
                
            except Exception as e:
                logger.error(f"Error en detección de caras: {e}")
                self.stats['processing_errors'] += 1
    
    def save_and_queue_worker(self, carpeta_capturas: str):
        """Worker para guardar imágenes y encolarlas para envío"""
        logger.info("Iniciando worker de guardado y encolado")
        
        while self.running:
            try:
                buffer_item = self.face_detection_buffer.get(timeout=1.0)
                if buffer_item is None:
                    continue
                
                # Generar nombre de archivo
                face_id = buffer_item.metadata.get('face_id', f'face_{int(time.time())}')
                filename = f"cara_{face_id}.jpg"
                filepath = os.path.join(carpeta_capturas, filename)
                
                # Guardar imagen
                success = cv2.imwrite(filepath, buffer_item.frame)
                
                if success:
                    logger.info(f"Cara guardada: {filepath}")
                    
                    # Añadir a cola de envío con prioridad basada en timestamp
                    priority = -buffer_item.timestamp  # Negativo para que más reciente = mayor prioridad
                    self.send_queue.put((priority, {
                        'filepath': filepath,
                        'timestamp': buffer_item.timestamp,
                        'metadata': buffer_item.metadata
                    }), block=False)
                else:
                    logger.error(f"Error al guardar imagen: {filepath}")
                
            except Exception as e:
                logger.error(f"Error en guardado y encolado: {e}")
                self.stats['processing_errors'] += 1
    
    def start_processing(self, carpeta_capturas: str, num_workers: int = 2):
        """Inicia el pipeline de procesamiento"""
        self.running = True
        
        # Crear directorio si no existe
        os.makedirs(carpeta_capturas, exist_ok=True)
        
        # Iniciar workers
        workers = [
            ('frame_processor', self.process_frames_worker),
            ('face_detector', self.face_detection_worker),
            ('saver_queuer', lambda: self.save_and_queue_worker(carpeta_capturas))
        ]
        
        # Crear múltiples workers para procesamiento intensivo
        for i in range(num_workers):
            for name, worker_func in workers:
                thread = threading.Thread(
                    target=worker_func,
                    name=f"{name}_{i}",
                    daemon=True
                )
                thread.start()
                self.threads.append(thread)
        
        logger.info(f"Pipeline iniciado con {len(self.threads)} workers")
    
    def stop_processing(self):
        """Detiene el pipeline de procesamiento"""
        logger.info("Deteniendo pipeline de procesamiento...")
        self.running = False
        
        # Esperar que terminen los hilos
        for thread in self.threads:
            thread.join(timeout=2.0)
        
        logger.info("Pipeline detenido")
    
    def get_pipeline_stats(self) -> dict:
        """Retorna estadísticas completas del pipeline"""
        return {
            'processing_stats': self.stats,
            'raw_buffer': self.raw_frame_buffer.get_stats(),
            'processed_buffer': self.processed_frame_buffer.get_stats(),
            'face_buffer': self.face_detection_buffer.get_stats(),
            'send_queue_size': self.send_queue.qsize(),
            'frame_counter': self.frame_counter,
            'face_counter': self.face_counter
        }

class CameraProcessor:
    def __init__(self, carpeta_capturas="capturas", ip_destino="192.168.1.100", puerto_servidor=8080, buffer_size=100):
        self.carpeta_capturas = carpeta_capturas
        self.ip_destino = ip_destino
        self.puerto_servidor = puerto_servidor
        
        # Pipeline de procesamiento con buffers
        self.pipeline = ProcessingPipeline(buffer_size)
        
        # Control de hilos
        self.running = False
        self.capture_thread = None
        self.sender_thread = None
        
        # Crear carpeta de capturas
        os.makedirs(carpeta_capturas, exist_ok=True)
        
        # Estadísticas
        self.stats_thread = None
        
    def stats_monitor_worker(self):
        """Worker para monitorear estadísticas del sistema"""
        while self.running:
            try:
                stats = self.pipeline.get_pipeline_stats()
                logger.info(f"Buffer Stats - Raw: {stats['raw_buffer']['current_size']}/{stats['raw_buffer']['max_size']}, "
                          f"Processed: {stats['processed_buffer']['current_size']}/{stats['processed_buffer']['max_size']}, "
                          f"Faces: {stats['face_buffer']['current_size']}/{stats['face_buffer']['max_size']}, "
                          f"Send Queue: {stats['send_queue_size']}")
                time.sleep(10)  # Mostrar stats cada 10 segundos
            except Exception as e:
                logger.error(f"Error en monitor de estadísticas: {e}")
    
    def capturar_desde_ip_camera(self, ip_camera, username=None, password=None, puerto_camara=80):
        """Captura frames desde cámara IP usando buffers"""
        stream_urls = [
            f"http://{ip_camera}:{puerto_camara}/video/mjpg.cgi",
            f"http://{ip_camera}:{puerto_camara}/mjpg/video.mjpg", 
            f"http://{ip_camera}:{puerto_camara}/videostream.cgi",
            f"http://{ip_camera}:{puerto_camara}/cgi-bin/mjpg/video.cgi",
            f"http://{ip_camera}:{puerto_camara}/video",
            f"http://{ip_camera}:{puerto_camara}/videostream.asf",
            f"http://{ip_camera}:{puerto_camara}/axis-cgi/mjpg/video.cgi",
            f"http://{ip_camera}:{puerto_camara}/mjpeg",
            f"http://{ip_camera}:{puerto_camara}/live",
            f"http://{ip_camera}:{puerto_camara}/stream",
            f"http://{ip_camera}:{puerto_camara}/cam/realmonitor",
            f"http://{ip_camera}:{puerto_camara}/video.mjpg",
            f"rtsp://{ip_camera}:{puerto_camara}/stream1",
            f"rtsp://{ip_camera}:{puerto_camara}/live/ch0",
            f"rtsp://{ip_camera}:{puerto_camara}/cam/realmonitor",
            f"rtsp://{ip_camera}:{puerto_camara}/live",
            f"http://{ip_camera}/video/mjpg.cgi",
            f"http://{ip_camera}/mjpg/video.mjpg"
        ]
        
        for url in stream_urls:
            try:
                logger.info(f"Intentando conectar a: {url}")
                
                if username and password:
                    url = url.replace("://", f"://{username}:{password}@")
                
                cap = cv2.VideoCapture(url)
                
                if cap.isOpened():
                    logger.info(f"Conectado exitosamente a {url}")
                    self.running = True
                    
                    # Iniciar pipeline de procesamiento
                    self.pipeline.start_processing(self.carpeta_capturas)
                    
                    frame_count = 0
                    while self.running:
                        ret, frame = cap.read()
                        
                        if ret:
                            # Añadir frame al pipeline con metadata
                            metadata = {
                                'source': 'ip_camera',
                                'url': url,
                                'frame_number': frame_count,
                                'capture_timestamp': time.time()
                            }
                            
                            success = self.pipeline.add_frame(frame, metadata)
                            if not success:
                                logger.warning("Buffer lleno, descartando frame")
                            
                            frame_count += 1
                            
                            # Mostrar frame (opcional)
                            if frame_count % 30 == 0:  # Mostrar cada 30 frames
                                cv2.imshow('IP Camera Feed', frame)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    self.running = False
                                    break
                                
                        else:
                            logger.error("Error al leer frame")
                            break
                        
                        time.sleep(0.033)  # ~30 FPS
                    
                    cap.release()
                    cv2.destroyAllWindows()
                    return True
                    
            except Exception as e:
                logger.error(f"Error con {url}: {e}")
                continue
        
        logger.error("No se pudo conectar a ningún stream de la cámara")
        return False
    
    def sender_worker(self):
        """Worker para enviar imágenes desde la cola"""
        logger.info(f"Iniciando worker de envío a {self.ip_destino}:{self.puerto_servidor}")
        
        while self.running:
            try:
                # Obtener elemento de la cola de envío
                if not self.pipeline.send_queue.empty():
                    priority, item = self.pipeline.send_queue.get(timeout=1.0)
                    
                    # Aquí iría la lógica de envío
                    # Por ahora solo loggeamos
                    logger.info(f"Enviando: {item['filepath']} (prioridad: {priority})")
                    
                    # Simular envío
                    time.sleep(0.1)
                else:
                    time.sleep(1.0)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error en worker de envío: {e}")
    
    def ejecutar_flujo_completo(self, ip_camera, username=None, password=None, puerto_camara=80):
        """Ejecuta el flujo completo con buffers"""
        logger.info("=== Iniciando flujo completo con buffers ===")
        logger.info(f"Cámara IP: {ip_camera}:{puerto_camara}")
        logger.info(f"Carpeta capturas: {self.carpeta_capturas}")
        logger.info(f"Servidor destino: {self.ip_destino}:{self.puerto_servidor}")
        
        # Iniciar workers auxiliares
        self.sender_thread = threading.Thread(target=self.sender_worker, daemon=True)
        self.sender_thread.start()
        
        self.stats_thread = threading.Thread(target=self.stats_monitor_worker, daemon=True)
        self.stats_thread.start()
        
        try:
            # Iniciar captura desde cámara IP
            self.capturar_desde_ip_camera(ip_camera, username, password, puerto_camara)
        except KeyboardInterrupt:
            logger.info("Deteniendo captura...")
            self.running = False
        finally:
            # Limpiar recursos
            self.pipeline.stop_processing()
            
            # Mostrar estadísticas finales
            final_stats = self.pipeline.get_pipeline_stats()
            logger.info(f"Estadísticas finales: {final_stats}")
        
        logger.info("Flujo completado")

def obtener_puerto_disponible(ip_destino, puerto_inicial=8080, max_puertos=20):
    """Encuentra un puerto disponible"""
    for i in range(max_puertos):
        puerto = puerto_inicial + i
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            resultado = sock.connect_ex((ip_destino, puerto))
            if resultado != 0:
                return puerto
    logger.warning("No se encontró un puerto disponible en el rango.")
    return puerto_inicial

def main():
    logger.info("=== Sistema de Captura y Envío con Buffers ===")
    print("1. Escanear red en busca de cámaras")
    print("2. Conectar a IP específica")
    print("3. Usar cámara local (webcam)")
    print("4. Modo de prueba con imágenes estáticas")
    
    opcion = input("Selecciona una opción (1-4): ")
    
    # Configuración
    IP_DESTINO = input("IP del servidor destino (default: 192.168.1.100): ") or "192.168.1.100"
    PUERTO_SERVIDOR = int(input("Puerto del servidor destino (default: 8080): ") or "8080")
    BUFFER_SIZE = int(input("Tamaño del buffer (default: 100): ") or "100")
    
    processor = CameraProcessor(
        carpeta_capturas="capturas",
        ip_destino=IP_DESTINO,
        puerto_servidor=PUERTO_SERVIDOR,
        buffer_size=BUFFER_SIZE
    )
    
    if opcion == "1":
        print("Escaneando red...")
        camaras = scan_network_for_cameras()
        if camaras:
            print(f"Cámaras encontradas: {camaras}")
            ip_seleccionada = input("Ingresa la IP de la cámara a usar: ")
            puerto_camara = int(input("Puerto de la cámara (default: 80): ") or "80")
            username = input("Usuario (opcional): ") or None
            password = input("Contraseña (opcional): ") or None
            processor.ejecutar_flujo_completo(ip_seleccionada, username, password, puerto_camara)
        else:
            print("No se encontraron cámaras")
    
    elif opcion == "2":
        ip_camera = input("Ingresa la IP de la cámara: ")
        puerto_camara = int(input("Puerto de la cámara (default: 80): ") or "80")
        username = input("Usuario (opcional): ") or None
        password = input("Contraseña (opcional): ") or None
        processor.ejecutar_flujo_completo(ip_camera, username, password, puerto_camara)
    
    elif opcion == "3":
        logger.info("Funcionalidad de cámara local pendiente de implementar con buffers")
        # Aquí se implementaría la captura desde cámara local usando el mismo sistema de buffers
    
    elif opcion == "4":
        logger.info("Modo de prueba - generando datos sintéticos")
        # Modo de prueba para verificar el funcionamiento de los buffers
        processor.pipeline.start_processing(processor.carpeta_capturas)
        
        # Generar frames sintéticos
        for i in range(50):
            synthetic_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            metadata = {'source': 'synthetic', 'frame_id': i}
            processor.pipeline.add_frame(synthetic_frame, metadata)
            time.sleep(0.1)
        
        time.sleep(5)  # Esperar procesamiento
        stats = processor.pipeline.get_pipeline_stats()
        logger.info(f"Estadísticas de prueba: {stats}")
        processor.pipeline.stop_processing()
    
    else:
        print("Opción no válida")

if __name__ == "__main__":
    main()
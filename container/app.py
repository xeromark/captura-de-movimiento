import threading
import time
import cv2
import os
from external_cam import capture_from_ip_camera, scan_network_for_cameras
from camera_photo import capturar_movimiento
from sender import monitorear_y_enviar
import queue
import socket
# Permite editar el puerto y buscar puertos disponibles a partir del 8080
def obtener_puerto_disponible(ip_destino, puerto_inicial=8080, max_puertos=20):
    for i in range(max_puertos):
        puerto = puerto_inicial + i
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            resultado = sock.connect_ex((ip_destino, puerto))
            if resultado != 0:  # Puerto no está en uso
                return puerto
    print("No se encontró un puerto disponible en el rango.")
    return puerto_inicial

# Modifica el main para usar el puerto disponible si el usuario lo desea
class CameraProcessor:
    def __init__(self, carpeta_capturas="capturas", ip_destino="192.168.1.100", puerto=8080):
        self.carpeta_capturas = carpeta_capturas
        self.ip_destino = ip_destino
        self.puerto = puerto
        self.frame_queue = queue.Queue(maxsize=10)
        self.running = False
        
        # Crear carpeta de capturas
        os.makedirs(carpeta_capturas, exist_ok=True)
        
        # Detector de caras para procesamiento
        self.detector_caras = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.contador = 0
        
    def procesar_frame_con_caras(self, frame):
        """Procesa un frame y guarda las caras detectadas"""
        if frame is None:
            return
            
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        caras = self.detector_caras.detectMultiScale(gris, 1.1, 4)
        
        if len(caras) > 0:
            for i, (x, y, w, h) in enumerate(caras):
                # Extraer solo el área de la cara
                cara_recortada = frame[y:y+h, x:x+w]
                
                # Guardar imagen de la cara
                nombre_archivo = f"{self.carpeta_capturas}/cara_{self.contador:05}_{i}.jpg"
                cv2.imwrite(nombre_archivo, cara_recortada)
                print(f"Cara detectada y guardada: {nombre_archivo}")
            
            self.contador += 1
            
    def capturar_desde_ip_camera(self, ip_camera, username=None, password=None):
        """Captura frames desde cámara IP y los procesa"""
        # URLs comunes para cámaras IP
        stream_urls = [
            f"http://{ip_camera}/video/mjpg.cgi",
            f"http://{ip_camera}/mjpg/video.mjpg",
            f"http://{ip_camera}/videostream.cgi",
            f"http://{ip_camera}:8080/video",
            f"rtsp://{ip_camera}/stream1"
        ]
        
        for url in stream_urls:
            try:
                print(f"Intentando conectar a: {url}")
                
                # Agregar autenticación si se proporciona
                if username and password:
                    url = url.replace("://", f"://{username}:{password}@")
                
                cap = cv2.VideoCapture(url)
                
                if cap.isOpened():
                    print(f"Conectado exitosamente a {url}")
                    self.running = True
                    
                    while self.running:
                        ret, frame = cap.read()
                        
                        if ret:
                            # Procesar frame para detectar caras
                            self.procesar_frame_con_caras(frame)
                            
                            # Mostrar frame (opcional)
                            cv2.imshow('IP Camera Feed', frame)
                            
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                self.running = False
                                break
                                
                        else:
                            print("Error al leer frame")
                            break
                        
                        time.sleep(0.1)  # Pequeña pausa
                    
                    cap.release()
                    cv2.destroyAllWindows()
                    return True
                    
            except Exception as e:
                print(f"Error con {url}: {e}")
                continue
        
        print("No se pudo conectar a ningún stream de la cámara")
        return False
    
    def iniciar_monitoreo_y_envio(self):
        """Inicia el monitoreo de la carpeta y envío automático"""
        def enviar_imagenes():
            print(f"Iniciando envío automático a {self.ip_destino}:{self.puerto}")
            monitorear_y_enviar(
                carpeta_capturas=self.carpeta_capturas,
                ip_destino=self.ip_destino,
                puerto=self.puerto,
                intervalo=5
            )
        
        # Ejecutar en hilo separado
        thread_envio = threading.Thread(target=enviar_imagenes, daemon=True)
        thread_envio.start()
        return thread_envio
    
    def ejecutar_flujo_completo(self, ip_camera, username=None, password=None):
        """Ejecuta el flujo completo: captura, procesamiento y envío"""
        print("=== Iniciando flujo completo ===")
        print(f"Cámara IP: {ip_camera}")
        print(f"Carpeta capturas: {self.carpeta_capturas}")
        print(f"Servidor destino: {self.ip_destino}:{self.puerto}")
        
        # Iniciar hilo de envío
        thread_envio = self.iniciar_monitoreo_y_envio()
        
        try:
            # Iniciar captura desde cámara IP
            self.capturar_desde_ip_camera(ip_camera, username, password)
        except KeyboardInterrupt:
            print("\nDeteniendo captura...")
            self.running = False
        
        print("Flujo completado")

def main():
    print("=== Sistema de Captura y Envío de Imágenes ===")
    print("1. Escanear red en busca de cámaras")
    print("2. Conectar a IP específica")
    print("3. Usar cámara local (webcam)")
    
    opcion = input("Selecciona una opción (1-3): ")
    
    # Configuración
    IP_DESTINO = input("IP del servidor destino (default: 192.168.1.100): ") or "192.168.1.100"
    PUERTO = int(input("Puerto del servidor (default: 8080): ") or "8080")
    
    processor = CameraProcessor(
        carpeta_capturas="capturas",
        ip_destino=IP_DESTINO,
        puerto=PUERTO
    )
    
    if opcion == "1":
        print("Escaneando red...")
        camaras = scan_network_for_cameras()
        if camaras:
            print(f"Cámaras encontradas: {camaras}")
            ip_seleccionada = input("Ingresa la IP de la cámara a usar: ")
            username = input("Usuario (opcional): ") or None
            password = input("Contraseña (opcional): ") or None
            processor.ejecutar_flujo_completo(ip_seleccionada, username, password)
        else:
            print("No se encontraron cámaras")
    
    elif opcion == "2":
        ip_camera = input("Ingresa la IP de la cámara: ")
        username = input("Usuario (opcional): ") or None
        password = input("Contraseña (opcional): ") or None
        processor.ejecutar_flujo_completo(ip_camera, username, password)
    
    elif opcion == "3":
        print("Usando cámara local...")
        
        # Definir callback personalizado para integrar con el sender
        def callback_foto_personalizado(imagen, contador, indice, carpeta, estado):
            nombre_archivo = f"{carpeta}/cara_{contador:05}_{indice}.jpg"
            cv2.imwrite(nombre_archivo, imagen)
            print(f"[{estado}] Cara guardada: {nombre_archivo}")
        
        # Iniciar envío automático
        thread_envio = processor.iniciar_monitoreo_y_envio()
        
        # Usar función de cámara local con callback personalizado
        capturar_movimiento(
            carpeta_capturas=processor.carpeta_capturas,
            callback_foto=callback_foto_personalizado
        )
    
    else:
        print("Opción no válida")

if __name__ == "__main__":
    main()
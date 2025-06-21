import os
import requests
import time
from pathlib import Path

def enviar_imagenes_a_ip(carpeta_capturas="capturas", ip_destino="192.168.1.100", puerto=8080, endpoint="/upload"):
    """
    Envía todas las imágenes de la carpeta de capturas a una IP específica.
    
    Args:
        carpeta_capturas: Carpeta donde están las imágenes
        ip_destino: IP del servidor destino
        puerto: Puerto del servidor
        endpoint: Endpoint para subir archivos
    """
    url = f"http://{ip_destino}:{puerto}{endpoint}"
    
    if not os.path.exists(carpeta_capturas):
        print(f"Error: La carpeta {carpeta_capturas} no existe")
        return
    
    # Obtener todas las imágenes
    extensiones = ('.jpg', '.jpeg', '.png', '.bmp')
    imagenes = [f for f in os.listdir(carpeta_capturas) 
                if f.lower().endswith(extensiones)]
    
    if not imagenes:
        print(f"No se encontraron imágenes en {carpeta_capturas}")
        return
    
    print(f"Enviando {len(imagenes)} imágenes a {url}")
    
    enviadas = 0
    errores = 0
    
    for imagen in imagenes:
        ruta_imagen = os.path.join(carpeta_capturas, imagen)
        
        try:
            with open(ruta_imagen, 'rb') as archivo:
                files = {'file': (imagen, archivo, 'image/jpeg')}
                response = requests.post(url, files=files, timeout=30)
                
                if response.status_code == 200:
                    print(f"✓ Enviada: {imagen}")
                    enviadas += 1
                else:
                    print(f"✗ Error enviando {imagen}: {response.status_code}")
                    errores += 1
                    
        except requests.exceptions.RequestException as e:
            print(f"✗ Error de conexión enviando {imagen}: {e}")
            errores += 1
        except Exception as e:
            print(f"✗ Error enviando {imagen}: {e}")
            errores += 1
        
        # Pequeña pausa entre envíos
        time.sleep(0.1)
    
    print(f"\nResumen: {enviadas} enviadas, {errores} errores")

def monitorear_y_enviar(carpeta_capturas="capturas", ip_destino="192.168.1.100", puerto=8080, intervalo=10):
    """
    Monitorea la carpeta y envía nuevas imágenes automáticamente.
    
    Args:
        carpeta_capturas: Carpeta a monitorear
        ip_destino: IP del servidor destino
        puerto: Puerto del servidor
        intervalo: Segundos entre verificaciones
    """
    enviadas_anteriormente = set()
    
    print(f"Monitoreando {carpeta_capturas} cada {intervalo} segundos...")
    print(f"Enviando a http://{ip_destino}:{puerto}/upload")
    print("Presiona Ctrl+C para detener")
    
    try:
        while True:
            if os.path.exists(carpeta_capturas):
                extensiones = ('.jpg', '.jpeg', '.png', '.bmp')
                imagenes_actuales = set(f for f in os.listdir(carpeta_capturas) 
                                      if f.lower().endswith(extensiones))
                
                nuevas_imagenes = imagenes_actuales - enviadas_anteriormente
                
                if nuevas_imagenes:
                    print(f"\nEncontradas {len(nuevas_imagenes)} nuevas imágenes")
                    
                    for imagen in nuevas_imagenes:
                        ruta_imagen = os.path.join(carpeta_capturas, imagen)
                        url = f"http://{ip_destino}:{puerto}/upload"
                        
                        try:
                            with open(ruta_imagen, 'rb') as archivo:
                                files = {'file': (imagen, archivo, 'image/jpeg')}
                                response = requests.post(url, files=files, timeout=30)
                                
                                if response.status_code == 200:
                                    print(f"✓ Enviada: {imagen}")
                                    enviadas_anteriormente.add(imagen)
                                else:
                                    print(f"✗ Error enviando {imagen}: {response.status_code}")
                                    
                        except Exception as e:
                            print(f"✗ Error enviando {imagen}: {e}")
            
            time.sleep(intervalo)
            
    except KeyboardInterrupt:
        print("\nMonitoreo detenido por el usuario")

if __name__ == "__main__":
    # Cambiar estos valores según tu configuración
    IP_SERVIDOR = "192.168.1.100"  # Cambiar por la IP destino
    PUERTO = 8080
    
    # Opción 1: Enviar todas las imágenes existentes
    # enviar_imagenes_a_ip(ip_destino=IP_SERVIDOR, puerto=PUERTO)
    
    # Opción 2: Monitorear y enviar automáticamente
    monitorear_y_enviar(ip_destino=IP_SERVIDOR, puerto=PUERTO)
import cv2
import time
import os

def capturar_movimiento(fps_bajo=1, fps_alto=25, umbral_movimiento=5000, 
                       tiempo_inactivo=5, carpeta_capturas="capturas", 
                       camara_id=0, callback_foto=None):
    """
    Captura imágenes solo cuando detecta caras, guardando únicamente el área de la cara.
    
    Args:
        fps_bajo: FPS cuando no hay movimiento
        fps_alto: FPS cuando hay movimiento
        umbral_movimiento: Sensibilidad de detección
        tiempo_inactivo: Segundos sin movimiento antes de modo bajo
        carpeta_capturas: Carpeta donde guardar imágenes
        camara_id: ID de la cámara a usar
        callback_foto: Función externa para procesar las fotos de caras
    """
    # Crear carpeta
    os.makedirs(carpeta_capturas, exist_ok=True)

    # Inicializar cámara
    camara = cv2.VideoCapture(camara_id)
    ret, frame_anterior = camara.read()
    if not ret:
        print("Error: No se pudo acceder a la cámara")
        return
        
    frame_anterior_gray = cv2.cvtColor(frame_anterior, cv2.COLOR_BGR2GRAY)
    frame_anterior_gray = cv2.GaussianBlur(frame_anterior_gray, (21, 21), 0)

    # Cargar detector de caras
    detector_caras = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    contador = 0
    modo_movimiento = False
    ultimo_movimiento = time.time()

    print("Capturando imágenes con caras. Presiona 'q' para salir.")

    try:
        while True:
            ret, frame_actual = camara.read()
            if not ret:
                break

            # Preprocesar
            gris = cv2.cvtColor(frame_actual, cv2.COLOR_BGR2GRAY)
            gris = cv2.GaussianBlur(gris, (21, 21), 0)

            # Detectar movimiento
            diferencia = cv2.absdiff(frame_anterior_gray, gris)
            _, umbral = cv2.threshold(diferencia, 25, 255, cv2.THRESH_BINARY)
            umbral = cv2.dilate(umbral, None, iterations=2)
            movimiento_detectado = cv2.countNonZero(umbral) > umbral_movimiento

            # Detectar caras
            caras = detector_caras.detectMultiScale(gris, 1.1, 4)
            cara_detectada = len(caras) > 0

            # Crear frame con rectangulos de caras
            frame_display = frame_actual.copy()
            
            # Mostrar estado
            texto_estado = "Movimiento" if movimiento_detectado else "Quieto"
            if cara_detectada:
                texto_estado += " + Cara"
                # Dibujar rectangulos alrededor de las caras
                for (x, y, w, h) in caras:
                    cv2.rectangle(frame_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            cv2.putText(frame_display, texto_estado, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0) if movimiento_detectado else (0, 0, 255), 2)
            cv2.imshow("Captura", frame_display)

            # Procesar caras detectadas
            if cara_detectada:
                for i, (x, y, w, h) in enumerate(caras):
                    # Extraer solo el área de la cara
                    cara_recortada = frame_actual[y:y+h, x:x+w]
                    
                    if callback_foto:
                        # Enviar imagen a función externa
                        callback_foto(cara_recortada, contador, i, carpeta_capturas, texto_estado)
                    else:
                        # Comportamiento por defecto
                        nombre_archivo = f"{carpeta_capturas}/cara_{contador:05}_{i}.jpg"
                        cv2.imwrite(nombre_archivo, cara_recortada)
                        print(f"[{texto_estado}] Cara guardada: {nombre_archivo}")
                
                contador += 1

            # Actualizar modo
            if movimiento_detectado:
                modo_movimiento = True
                ultimo_movimiento = time.time()
                time.sleep(1 / fps_alto)
            else:
                if time.time() - ultimo_movimiento > tiempo_inactivo:
                    modo_movimiento = False
                time.sleep(1 / (fps_alto if modo_movimiento else fps_bajo))

            # Actualizar referencia
            frame_anterior_gray = gris

            # Salida con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nCaptura interrumpida por el usuario.")
    finally:
        camara.release()
        cv2.destroyAllWindows()

def procesar_foto_default(imagen, contador, indice, carpeta, estado):
    """Función por defecto para procesar fotos"""
    nombre_archivo = f"{carpeta}/cara_{contador:05}_{indice}.jpg"
    cv2.imwrite(nombre_archivo, imagen)
    print(f"[{estado}] Cara guardada: {nombre_archivo}")

# Para ejecutar la función
if __name__ == "__main__":
    capturar_movimiento()

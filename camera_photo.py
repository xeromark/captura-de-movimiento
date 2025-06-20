import cv2
import time
import os

# Configuración
FPS_BAJO = 1
FPS_ALTO = 25
UMBRAL_MOVIMIENTO = 5000  # sensibilidad
TIEMPO_INACTIVO = 5       # segundos sin movimiento antes de volver a modo bajo

# Crear carpeta
os.makedirs("capturas", exist_ok=True)

# Inicializar cámara
camara = cv2.VideoCapture(0)
ret, frame_anterior = camara.read()
frame_anterior_gray = cv2.cvtColor(frame_anterior, cv2.COLOR_BGR2GRAY)
frame_anterior_gray = cv2.GaussianBlur(frame_anterior_gray, (21, 21), 0)

contador = 0
modo_movimiento = False
ultimo_movimiento = time.time()

print("Capturando imágenes. Presiona 'q' para salir.")

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
        movimiento_detectado = cv2.countNonZero(umbral) > UMBRAL_MOVIMIENTO

        # Mostrar estado
        texto_estado = "Movimiento" if movimiento_detectado else "Quieto"
        cv2.putText(frame_actual, texto_estado, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if movimiento_detectado else (0, 0, 255), 2)
        cv2.imshow("Captura", frame_actual)

        # Guardar imagen
        nombre_archivo = f"capturas/img_{contador:05}.jpg"
        cv2.imwrite(nombre_archivo, frame_actual)
        print(f"[{texto_estado}] Imagen guardada: {nombre_archivo}")
        contador += 1

        # Actualizar modo
        if movimiento_detectado:
            modo_movimiento = True
            ultimo_movimiento = time.time()
            time.sleep(1 / FPS_ALTO)
        else:
            if time.time() - ultimo_movimiento > TIEMPO_INACTIVO:
                modo_movimiento = False
            time.sleep(1 / (FPS_ALTO if modo_movimiento else FPS_BAJO))

        # Actualizar referencia
        frame_anterior_gray = gris

        # Salida con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nCaptura interrumpida por el usuario.")

camara.release()
cv2.destroyAllWindows()

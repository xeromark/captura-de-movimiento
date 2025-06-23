import cv2
import time

# Inicializa la cámara (por lo general, 0 es la cámara predeterminada)
camara = cv2.VideoCapture(0)

# Configura el tamaño del video
ancho = int(camara.get(cv2.CAP_PROP_FRAME_WIDTH))
alto = int(camara.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define el códec y crea el objeto VideoWriter
video_salida = cv2.VideoWriter('salida.avi', cv2.VideoWriter_fourcc(*'XVID'), 2, (ancho, alto))

print("Grabando video a 2 FPS... Presiona 'q' para salir.")

try:
    while True:
        ret, frame = camara.read()
        if not ret:
            print("No se pudo capturar el frame.")
            break
        
        video_salida.write(frame)
        cv2.imshow("Grabando...", frame)
        
        # Espera 0.5 segundos para grabar a 2 FPS
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\nInterrumpido por el usuario.")

# Libera recursos
camara.release()
video_salida.release()
cv2.destroyAllWindows()

import argparse
import threading
import sys
import os
import time
import base64
import tempfile
import cv2
import queue
import math

ROOT = os.path.dirname(__file__)

import torch
import psycopg2
from dotenv import load_dotenv
from urllib.parse import urlparse
from signhandler.siamese_network import SiameseNetwork
from signhandler.signer import generate_face_signature, capture_square_photo, FaceEmbeddingGenerator
from signhandler.comparator import SignatureComparator

# Cargar variables de entorno
load_dotenv()

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
                        nombre VARCHAR(100) NOT NULL,
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
    def __init__(self, model_path, device='cpu', face_threshold=0.85, distance_threshold=0.45):
        self.model_path = model_path
        self.device = device
        self.face_threshold = face_threshold
        self.distance_threshold = distance_threshold
        self.min_similarity_percentage = 85.0  # Nivel de exigencia al 85%
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

        self.comparator = SignatureComparator(self.model_path, device=self.device)
        self.embedding_generator = FaceEmbeddingGenerator(self.model_path, device=self.device)
        self.detector_caras = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.firmas_db = self.obtener_firmas_db()
        self.embeddings_cache = {}
        print(f"📊 {len(self.firmas_db)} firmas cargadas desde PostgreSQL")
        self._precalcular_embeddings()

    def obtener_firmas_db(self):
        """Lee todas las firmas válidas de la base de datos PostgreSQL."""
        try:
            with psycopg2.connect(**DB_PARAMS) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT id, nombre, firma FROM firmas WHERE firma IS NOT NULL AND LENGTH(firma) > 0")
                    resultados = cur.fetchall()
                    
                    # Filtrar firmas válidas (base64)
                    firmas_validas = []
                    for id_firma, nombre, firma in resultados:
                        try:
                            if firma and len(firma.strip()) > 0:
                                base64.b64decode(firma)  # Validar que es base64
                                firmas_validas.append({
                                    'id': id_firma,
                                    'nombre': nombre,
                                    'firma': firma
                                })
                        except Exception:
                            continue  # Ignorar firmas corruptas
                    
                    return firmas_validas
        except Exception as e:
            print(f"Error al conectar a la base de datos: {e}")
            return []

    def insertar_firma(self, firma, nombre="Desconocido"):
        """Inserta una nueva firma en la base de datos PostgreSQL."""
        try:
            # Validar que la firma no esté vacía
            if not firma or len(firma.strip()) == 0:
                print("❌ Error: Firma vacía, no se guardará")
                return False
                
            # Validar formato base64
            try:
                base64.b64decode(firma)
            except Exception:
                print("❌ Error: Firma no es base64 válido, no se guardará")
                return False
            
            with psycopg2.connect(**DB_PARAMS) as conn:
                with conn.cursor() as cur:
                    cur.execute("INSERT INTO firmas (nombre, firma) VALUES (%s, %s) RETURNING id", (nombre, firma))
                    nuevo_id = cur.fetchone()[0]
                conn.commit()
            print(f"✓ Firma guardada en PostgreSQL para '{nombre}' (ID: {nuevo_id})")
            # Actualizar lista de firmas
            self.firmas_db = self.obtener_firmas_db()
            # Recalcular embeddings
            self._precalcular_embeddings()
            return True
        except Exception as e:
            print(f"Error al guardar firma: {e}")
            return False

    def comparar_firma_con_db(self, firma_nueva):
        """
        Compara una firma con todas las almacenadas en la DB usando lógica exacta de Discord.
        
        Implementa:
        - distance = torch.nn.functional.pairwise_distance(output1, output2).item()
        - if (distances < threshold): conocido = true
        """
        if not self.firmas_db:
            return None, float('inf'), False, 0.0, "Desconocido"
        
        min_distance = float('inf')
        max_distance = -float('inf')
        distancias = []
        firma_mas_similar = None
        nombre_mas_similar = "Desconocido"
        
        for item in self.firmas_db:
            try:
                firma_db = item['firma']
                nombre_db = item['nombre']
                
                # Validar que la firma no esté vacía
                if not firma_db or len(firma_db.strip()) == 0:
                    continue
                
                # Usar lógica exacta de Discord: compare_with_discord_logic
                distance = self.comparator.compare_with_discord_logic(firma_nueva, firma_db)
                distancias.append(distance)
                
                if distance < min_distance:
                    min_distance = distance
                    firma_mas_similar = firma_db
                    nombre_mas_similar = nombre_db
                    
                if distance > max_distance:
                    max_distance = distance
                    
            except Exception as e:
                # Solo mostrar el primer error para evitar spam
                if len(distancias) == 0:
                    print(f"⚠️ Error procesando firmas: {e}")
                continue
        
        # 📊 Estadísticas para debugging
        if distancias:
            promedio = sum(distancias) / len(distancias)
            print(f"📊 Discord Logic - Min: {min_distance:.3f}, Max: {max_distance:.3f}, Promedio: {promedio:.3f}, Total: {len(distancias)}")
        
        # Calcular porcentaje de similitud (inverso de la distancia normalizada)
        if min_distance == float('inf'):
            similarity_percentage = 0.0
        else:
            # Normalizar distancia a un porcentaje (0-100%)
            # Distancia 0 = 100% similitud, distancia alta = 0% similitud
            similarity_percentage = max(0, 100 * math.exp(-min_distance / 2))
        
        # Lógica simplificada: SOLO VERIFICACIÓN POR PORCENTAJE DE SIMILITUD >= 85%
        conocido = similarity_percentage >= self.min_similarity_percentage
        
        # 📊 Debug simplificado para el nuevo sistema
        print(f"🔍 SEGURIDAD 85% - Similitud: {similarity_percentage:.1f}% >= {self.min_similarity_percentage:.1f}% = {conocido}")
        if conocido:
            print(f"👤 Persona identificada: {nombre_mas_similar}")
        
        return firma_mas_similar, min_distance, conocido, similarity_percentage, nombre_mas_similar

    def _precalcular_embeddings(self):
        """Pre-calcula embeddings de todas las firmas de la BD para comparación más rápida."""
        if not self.comparator or not self.firmas_db:
            return
        
        print("🔄 Pre-calculando embeddings de firmas de BD...")
        self.embeddings_cache = {}
        
        for i, item in enumerate(self.firmas_db):
            try:
                firma = item['firma']
                nombre = item['nombre']
                # Generar embedding para esta firma
                embedding = self.comparator.get_embedding(firma)
                self.embeddings_cache[i] = {
                    'nombre': nombre,
                    'firma': firma,
                    'embedding': embedding
                }
            except Exception as e:
                print(f"⚠️ Error calculando embedding para firma {i} ({item.get('nombre', 'Sin nombre')}): {e}")
                continue
        
        print(f"✅ Pre-calculados {len(self.embeddings_cache)} embeddings")

    def iniciar_captura_webcam(self):
        """Inicia la captura desde la webcam local."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("No se pudo acceder a la webcam.")
            return False

        print("Webcam conectada. Presiona 'espacio' para guardar firma, 'q' para salir.")
        print("Al guardar una firma, se te pedirá el nombre de la persona.")
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
                if self.comparator:
                    cara = frame[y:y+h, x:x+w]
                    embedding = self.embedding_generator.generate_embedding(cara)
                    _, distance, is_known, similarity_percentage, nombre = self.comparar_firma_con_db(embedding)

                    # Determinar color del rectángulo basado en si es conocido
                    color = (0, 255, 0) if is_known else (0, 0, 255)  # Verde para conocido, rojo para desconocido
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Mostrar nombre y porcentaje de similitud
                    if is_known:
                        texto = f"{nombre} {similarity_percentage:.1f}%"
                        status = "CONOCIDO"
                    else:
                        texto = f"DESCONOCIDO {similarity_percentage:.1f}%"
                        status = "DESCONOCIDO"
                    
                    cv2.putText(frame, texto, (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                color, 2)
                    
                    # Mostrar información adicional debajo
                    texto_detalle = f"d={distance:.2f}"
                    cv2.putText(frame, texto_detalle, (x, y+h+20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                                color, 1)
                    
                    # 🐛 DEBUG: Mostrar valores para diagnóstico
                    print(f"🔍 DEBUG - Distancia: {distance:.3f}, Similitud: {similarity_percentage:.1f}%, Conocido: {is_known}, Nombre: {nombre}")
                else:
                    # Si no hay comparador, solo dibujar rectángulo verde
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Mostrar frame
            cv2.imshow("Captura de Caras", frame)
            
            # Comprobar teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
                break
            elif key == 32:  # Tecla espacio
                if len(caras) > 0:
                    # Tomar la primera cara detectada
                    x, y, w, h = caras[0]
                    cara = frame[y:y+h, x:x+w]
                    embedding = self.embedding_generator.generate_embedding(cara)
                    
                    # Pedir nombre al usuario
                    print("\n" + "="*50)
                    print("💾 GUARDANDO NUEVA FIRMA")
                    print("="*50)
                    nombre = input("Introduce el nombre de la persona: ").strip()
                    if not nombre:
                        nombre = "Desconocido"
                    
                    if self.insertar_firma(embedding, nombre):
                        # Guardar imagen de la cara
                        timestamp = int(time.time())
                        img_path = os.path.join(self.carpeta_capturas, f"cara_{nombre}_{timestamp}.jpg")
                        cv2.imwrite(img_path, cara)
                        print(f"✅ Imagen guardada en {img_path}")
                        print(f"✅ Firma de '{nombre}' registrada correctamente")
                    else:
                        print("❌ Error al guardar la firma")
                    print("="*50)
                else:
                    print("⚠️ No se detectó ninguna cara para guardar")

        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--face-threshold", type=float, default=0.7)
    parser.add_argument("--distance-threshold", type=float, default=0.45)
    args = parser.parse_args()

    sistema = IntegratedSystem(
        model_path=os.path.join(ROOT, "signhandler", "model.pth"),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        face_threshold=args.face_threshold,
        distance_threshold=args.distance_threshold
    )

    sistema.iniciar_captura_webcam()

if __name__ == "__main__":
    main()
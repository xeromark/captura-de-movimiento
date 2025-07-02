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
sys.path.insert(0, os.path.join(ROOT, "..", "signhandler"))

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
            'dbname': os.getenv('DB_NAME', 'postgres'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'password'),
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '5432'))
        }

# Parámetros de BD globales
DB_PARAMS = get_db_params()

def crear_tabla_personas():
    """Crea la tabla de personas con nombres si no existe"""
    try:
        with psycopg2.connect(**DB_PARAMS) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS personas (
                        id SERIAL PRIMARY KEY,
                        nombre VARCHAR(255) NOT NULL,
                        firma TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
                print("✅ Tabla 'personas' verificada/creada")
    except Exception as e:
        print(f"⚠️ Error creando tabla de personas: {e}")

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
    def __init__(self, model_path, device='cpu', face_threshold=0.7, distance_threshold=0.45):
        self.model_path = model_path
        self.device = device
        self.face_threshold = face_threshold  # Threshold para detección de rostros
        self.distance_threshold = distance_threshold  # Threshold para comparación de firmas (≈85% similitud)
        self.min_similarity_percentage = 85.0  # Porcentaje mínimo requerido para reconocimiento
        self.running = False
        self.frame_queue = queue.Queue(maxsize=10)
        # YA NO CREAMOS CARPETA DE CAPTURAS - Solo guardamos firmas en BD
        # self.carpeta_capturas = os.path.join(ROOT, "capturas")
        # os.makedirs(self.carpeta_capturas, exist_ok=True)

        # Verificar conexión BD y crear tabla si es necesario
        print(f"📋 Conectando a BD: {DB_PARAMS['host']}:{DB_PARAMS['port']}")
        if probar_conexion_bd():
            crear_tabla_personas()
        else:
            print("⚠️ Continuando sin conexión a BD")

        # Cargar detector de caras
        self.detector_caras = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print(f"🎯 Threshold de detección facial: {self.face_threshold}")
        print(f"📏 Threshold de distancia de firmas: {self.distance_threshold}")
        
        # Cargar firmas de la base de datos
        self.firmas_db = self.obtener_firmas_db()
        self.embeddings_cache = {}  # Cache para embeddings de firmas de BD
        self.ultimo_refresh_db = time.time()
        self.refresh_interval = 30  # Refrescar BD cada 30 segundos
        print(f"📊 Se cargaron {len(self.firmas_db)} firmas de la base de datos")
        
        # Pre-calcular embeddings de firmas existentes para comparación más rápida
        self._precalcular_embeddings()

        # Cargar el comparador
        try:
            self.comparator = SignatureComparator(self.model_path, device=self.device)
            self.embedding_generator = FaceEmbeddingGenerator(self.model_path, device=self.device)
            print(f"✅ Modelo cargado correctamente desde {self.model_path}")
        except Exception as e:
            print(f"❌ Error al cargar el modelo: {e}")
            print(f"📋 Detalles del error: {type(e).__name__}")
            self.comparator = None
            self.embedding_generator = None

    def obtener_firmas_db(self):
        """Lee todas las firmas válidas de la base de datos."""
        try:
            with psycopg2.connect(**DB_PARAMS) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT id, nombre, firma FROM personas WHERE firma IS NOT NULL AND LENGTH(firma) > 0")
                    resultados = cur.fetchall()
                    
                    # Filtrar firmas válidas (base64)
                    firmas_validas = []
                    for id_persona, nombre, firma in resultados:
                        try:
                            if firma and len(firma.strip()) > 0:
                                base64.b64decode(firma)  # Validar que es base64
                                firmas_validas.append({
                                    'id': id_persona,
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
        """Inserta una nueva firma en la base de datos con nombre."""
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
                    cur.execute("INSERT INTO personas (nombre, firma) VALUES (%s, %s) RETURNING id", 
                              (nombre, firma))
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
        Compara una firma con todas las almacenadas en la DB.
        """
        if not self.firmas_db:
            return None, float('inf'), False, 0.0, "Desconocido"
        
        min_distance = float('inf')
        max_distance = -float('inf')
        distancias = []
        persona_mas_similar = None
        
        for persona in self.firmas_db:
            try:
                # Validar que la firma no esté vacía
                firma_db = persona['firma']
                if not firma_db or len(firma_db.strip()) == 0:
                    continue
                
                # Usar lógica exacta de Discord: compare_with_discord_logic
                distance = self.comparator.compare_with_discord_logic(firma_nueva, firma_db)
                distancias.append(distance)
                
                if distance < min_distance:
                    min_distance = distance
                    persona_mas_similar = persona
                    
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
        
        # Obtener nombre de la persona más similar
        nombre = persona_mas_similar['nombre'] if persona_mas_similar and conocido else "Desconocido"
        
        # 📊 Debug simplificado para el nuevo sistema
        print(f"🔍 SEGURIDAD 85% - Similitud: {similarity_percentage:.1f}% >= {self.min_similarity_percentage:.1f}% = {conocido}")
        print(f"👤 Persona identificada: {nombre}")
        
        return persona_mas_similar, min_distance, conocido, similarity_percentage, nombre

    def _precalcular_embeddings(self):
        """Pre-calcula embeddings de todas las firmas de la BD para comparación más rápida."""
        if not self.comparator or not self.firmas_db:
            return
        
        print("🔄 Pre-calculando embeddings de firmas de BD...")
        self.embeddings_cache = {}
        
        for i, persona in enumerate(self.firmas_db):
            try:
                # Generar embedding para esta firma
                embedding = self.comparator.get_embedding(persona['firma'])
                self.embeddings_cache[i] = {
                    'id': persona['id'],
                    'nombre': persona['nombre'],
                    'firma': persona['firma'],
                    'embedding': embedding
                }
            except Exception as e:
                print(f"⚠️ Error calculando embedding para persona {persona.get('nombre', 'N/A')}: {e}")
                continue
        
        print(f"✅ Pre-calculados {len(self.embeddings_cache)} embeddings")

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

            for (x, y, w, h) in caras:
                # Procesar cara solo si tenemos comparador
                if self.comparator:
                    cara = frame[y:y+h, x:x+w]
                    embedding = self.embedding_generator.generate_embedding(cara)
                    _, distance, is_known, similarity_percentage, nombre = self.comparar_firma_con_db(embedding)

                    # Determinar color del rectángulo basado en si es conocido
                    color = (0, 255, 0) if is_known else (0, 0, 255)  # Verde para conocido, rojo para desconocido
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Mostrar nombre y porcentaje de similitud
                    status = "CONOCIDO" if is_known else "DESCONOCIDO"
                    texto = f"{nombre} - {status} {similarity_percentage:.1f}%"
                    
                    # 🐛 DEBUG: Mostrar valores para diagnóstico
                    print(f"🔍 DEBUG - Distancia: {distance:.3f}, Similitud: {similarity_percentage:.1f}%, Conocido: {is_known}, Nombre: {nombre}")
                    
                    cv2.putText(frame, texto, (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                color, 2)
                    
                    # Mostrar información adicional debajo
                    texto_detalle = f"d={distance:.2f}"
                    cv2.putText(frame, texto_detalle, (x, y+h+20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                                color, 1)

                    # Guardar firma al presionar espacio
                    key = cv2.waitKey(1) & 0xFF
                    if key == 32:  # Tecla espacio
                        # Pedir nombre para nueva persona
                        print("Ingresa el nombre de la persona:")
                        nombre_nuevo = input().strip() or "Desconocido"
                        
                        if self.insertar_firma(embedding, nombre_nuevo):
                            print(f"✓ Firma guardada para '{nombre_nuevo}' - Similitud: {similarity_percentage:.1f}%")
                            # YA NO GUARDAMOS IMAGEN FÍSICA - Solo firma en BD
                            print(f"📊 Firma procesada y guardada en PostgreSQL")
                else:
                    # Si no hay comparador, solo dibujar rectángulo verde
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

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

    def _refresh_db_if_needed(self):
        """Refresca las firmas de la BD si ha pasado suficiente tiempo."""
        current_time = time.time()
        if current_time - self.ultimo_refresh_db > self.refresh_interval:
            old_count = len(self.firmas_db)
            self.firmas_db = self.obtener_firmas_db()
            new_count = len(self.firmas_db)
            
            if new_count != old_count:
                print(f"🔄 BD actualizada: {old_count} -> {new_count} firmas")
                self._precalcular_embeddings()
            
            self.ultimo_refresh_db = current_time

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

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
import json

ROOT = os.path.dirname(__file__)
DB_JSON = os.path.join(ROOT, "firmas.json")

import torch
from signhandler.siamese_network import SiameseNetwork
from signhandler.signer import generate_face_signature, capture_square_photo, FaceEmbeddingGenerator
from signhandler.comparator import SignatureComparator

class IntegratedSystem:
    def __init__(self, model_path, device='cpu', face_threshold=0.7, distance_threshold=0.45):
        self.model_path = model_path
        self.device = device
        self.face_threshold = face_threshold
        self.distance_threshold = distance_threshold
        self.min_similarity_percentage = 80.0
        self.running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.carpeta_capturas = os.path.join(ROOT, "capturas")
        os.makedirs(self.carpeta_capturas, exist_ok=True)

        self.comparator = SignatureComparator(self.model_path, device=self.device)
        self.embedding_generator = FaceEmbeddingGenerator(self.model_path, device=self.device)
        self.detector_caras = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.firmas_db = self.obtener_firmas_db()
        self.embeddings_cache = {}
        print(f"📊 {len(self.firmas_db)} firmas cargadas desde {DB_JSON}")
        self._precalcular_embeddings()

    def obtener_firmas_db(self):
        if not os.path.exists(DB_JSON):
            with open(DB_JSON, "w") as f:
                json.dump([], f)
            return []

        with open(DB_JSON, "r") as f:
            datos = json.load(f)

        firmas_validas = []
        for item in datos:
            try:
                base64.b64decode(item['firma'])
                firmas_validas.append(item['firma'])
            except Exception:
                continue
        return firmas_validas

    def insertar_firma(self, firma):
        if not firma or len(firma.strip()) == 0:
            print("❌ Firma vacía, no se guardará")
            return False
        try:
            base64.b64decode(firma)
            with open(DB_JSON, "r") as f:
                datos = json.load(f)
            nuevo_id = max([item['id'] for item in datos], default=0) + 1
            datos.append({"id": nuevo_id, "firma": firma})
            with open(DB_JSON, "w") as f:
                json.dump(datos, f, indent=4)
            print(f"✓ Firma guardada en {DB_JSON} (ID: {nuevo_id})")
            self.firmas_db = self.obtener_firmas_db()
            self._precalcular_embeddings()
            return True
        except Exception as e:
            print(f"⚠️ Error al guardar firma: {e}")
            return False

    def comparar_firma_con_db(self, firma_nueva):
        if not self.firmas_db:
            return None, float('inf'), False, 0.0
        min_distance = float('inf')
        firma_mas_similar = None

        for firma_db in self.firmas_db:
            try:
                distance = self.comparator.compare_with_discord_logic(firma_nueva, firma_db)
                if distance < min_distance:
                    min_distance = distance
                    firma_mas_similar = firma_db
            except Exception:
                continue

        similarity_percentage = max(0, 100 * math.exp(-min_distance / 2))
        conocido = min_distance < self.distance_threshold and similarity_percentage >= self.min_similarity_percentage
        return firma_mas_similar, min_distance, conocido, similarity_percentage

    def _precalcular_embeddings(self):
        self.embeddings_cache = {}
        for i, firma in enumerate(self.firmas_db):
            try:
                embedding = self.comparator.get_embedding(firma)
                self.embeddings_cache[i] = {'firma': firma, 'embedding': embedding}
            except Exception:
                continue

    def iniciar_captura_webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("No se pudo acceder a la webcam.")
            return False

        print("Presiona 'espacio' para guardar firma, 'q' para salir.")
        self.running = True

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            caras = self.detector_caras.detectMultiScale(gris, 1.1, 4)

            for (x, y, w, h) in caras:
                cara = frame[y:y+h, x:x+w]
                embedding = self.embedding_generator.generate_embedding(cara)
                _, distance, is_known, similarity_percentage = self.comparar_firma_con_db(embedding)

                color = (0, 255, 0) if is_known else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                texto = f"{'CONOCIDO' if is_known else 'DESCONOCIDO'} {similarity_percentage:.1f}%"
                cv2.putText(frame, texto, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                key = cv2.waitKey(1) & 0xFF
                if key == 32:
                    firma_b64 = embedding
                    if self.insertar_firma(firma_b64):
                        img_path = os.path.join(self.carpeta_capturas, f"cara_{time.time()}.jpg")
                        cv2.imwrite(img_path, cara)
                        print(f"Imagen guardada en {img_path}")

            cv2.imshow("Captura", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

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

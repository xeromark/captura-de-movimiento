import os
import json
import math
import base64
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, Response

from signhandler.signer import FaceEmbeddingGenerator
from signhandler.comparator import SignatureComparator
from main import IntegratedSystem

ROOT = os.path.dirname(__file__)
DB_JSON = os.path.join(ROOT, "firmas.json")

app = Flask(__name__)

# Inicializa el sistema principal con modelo ya preparado
sistema = IntegratedSystem(
    model_path=os.path.join(ROOT, "signhandler", "model.pth"),
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

@app.route('/video_feed')
def video_feed():
    def gen():
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detección y dibujo
            gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            caras = sistema.detector_caras.detectMultiScale(gris, 1.1, 4)

            for (x, y, w, h) in caras:
                cara = frame[y:y+h, x:x+w]
                firma = sistema.embedding_generator.generate_embedding(cara)
                _, _, conocido, sim = sistema.comparar_firma_con_db(firma)
                color = (0, 255, 0) if conocido else (0, 0, 255)
                texto = f"{'Conocido' if conocido else 'Desconocido'} {sim:.1f}%"
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, texto, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_b64 = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_b64 + b'\r\n')

        cap.release()

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/guardar_firma', methods=['POST'])
def guardar_firma():
    data = request.get_json()
    img_b64 = data.get('imagen')
    nombre = data.get('nombre')

    if not img_b64 or not nombre:
        return jsonify({"status": "error", "error": "Faltan datos"}), 400

    try:
        img_bytes = base64.b64decode(img_b64.split(",")[1])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        caras = sistema.detector_caras.detectMultiScale(gris, 1.1, 4)

        if len(caras) == 0:
            return jsonify({"status": "error", "error": "No se detectó rostro"}), 422

        x, y, w, h = caras[0]
        cara = img[y:y+h, x:x+w]

        firma_b64 = sistema.embedding_generator.generate_embedding(cara)

        if sistema.insertar_firma(firma_b64, nombre):
            return jsonify({"status": "ok", "firma": firma_b64}), 200
        else:
            return jsonify({"status": "error", "error": "No se pudo guardar firma"}), 500

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/comparar', methods=['POST'])
def comparar():
    data = request.get_json()
    img_b64 = data.get('imagen')

    if not img_b64:
        return jsonify({"error": "Falta imagen"}), 400

    try:
        img_bytes = base64.b64decode(img_b64.split(",")[1])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        caras = sistema.detector_caras.detectMultiScale(gris, 1.1, 4)

        if len(caras) == 0:
            return jsonify({"error": "No se detectó rostro"}), 422

        x, y, w, h = caras[0]
        cara = img[y:y+h, x:x+w]

        firma_b64 = sistema.embedding_generator.generate_embedding(cara)

        _, distance, conocido, similarity = sistema.comparar_firma_con_db(firma_b64)

        return jsonify({"firma": firma_b64, "conocido": conocido, "similaridad": similarity}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generar_firma', methods=['POST'])
def generar_firma():
    data = request.get_json()
    img_b64 = data.get('imagen')

    if not img_b64:
        return jsonify({"error": "Falta imagen"}), 400

    try:
        img_bytes = base64.b64decode(img_b64.split(",")[1])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        caras = sistema.detector_caras.detectMultiScale(gris, 1.1, 4)

        if len(caras) == 0:
            return jsonify({"error": "No se detectó rostro"}), 422

        x, y, w, h = caras[0]
        cara = img[y:y+h, x:x+w]

        firma_b64 = sistema.embedding_generator.generate_embedding(cara)

        return jsonify({"firma": firma_b64}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
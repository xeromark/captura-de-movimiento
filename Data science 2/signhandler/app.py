# filepath: /home/ignatus/Documentos/Github/captura-de-movimiento/signhandler/app.py
import os
import base64
import tempfile

import torch
import psycopg2
from flask import Flask, request, jsonify
import torch.serialization

# Importar SiameseNetwork para que esté disponible en el contexto
from signhandler.siamese_network import SiameseNetwork
from signer import sign_image, generate_keys
from comparator import SignatureComparator
from dbhdlr import db_params

# Configuración
HOST       = "0.0.0.0"
PORT       = 5000
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pth")
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

app = Flask(__name__)
comparator = SignatureComparator(MODEL_PATH, device=DEVICE)
priv_key, pub_key = generate_keys()  # claves temporales para firmar

def obtener_firmas_db():
    """Lee todas las firmas (raw) de la base de datos."""
    with psycopg2.connect(**db_params) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT firma FROM firmas")
            return [row[0] for row in cur.fetchall()]

@app.route('/comparar', methods=['POST'])
def comparar_firma():
    data = request.get_json(silent=True) or {}
    foto_b64 = data.get('foto')
    if not foto_b64:
        return jsonify({"error": "Falta la foto en base64"}), 400

    # Decodifica y guarda la imagen en un tmp file
    try:
        img_bytes = base64.b64decode(foto_b64)
    except Exception:
        return jsonify({"error": "Base64 inválido"}), 400

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    try:
        tmp.write(img_bytes)
        tmp.flush()
        firma_nueva = sign_image(tmp.name, priv_key)
    finally:
        tmp.close()
        os.unlink(tmp.name)

    # Trae las firmas almacenadas
    firmas_db = obtener_firmas_db()
    if not firmas_db:
        return jsonify({"error": "No hay firmas en la base de datos"}), 404

    # Compara cada firma y busca la máxima similitud
    max_sim = -1.0
    firma_mas_similar = None
    for firma_db in firmas_db:
        try:
            sim = comparator.compare(firma_nueva, firma_db)
        except Exception:
            continue
        if sim > max_sim:
            max_sim = sim
            firma_mas_similar = firma_db

    if firma_mas_similar is None:
        return jsonify({"error": "No se pudo comparar firmas"}), 500

    return jsonify({
        "firma_mas_similar": firma_mas_similar,
        "similitud":          round(max_sim * 100, 2)
    })

if __name__ == "__main__":
    app.run(host=HOST, port=PORT)

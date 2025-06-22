from flask import Flask, request, jsonify
import tempfile
import base64
import os
from signer import sign_image, generate_keys
from comparator import SignatureComparator
from dbhdlr import db_params
import psycopg2

# filepath: /home/ignatus/Documentos/Github/captura-de-movimiento/signhandler/app.py


# Configuración
HOST = "0.0.0.0"
PORT = 5000
MODEL_PATH = "modelo.pkl"  # Cambia esto por la ruta real de tu modelo

# Inicializa Flask y el comparador
app = Flask(__name__)
comparator = SignatureComparator(MODEL_PATH)
priv_key, pub_key = generate_keys()  # Genera claves temporales para firmar

def obtener_firmas_db():
    """Obtiene todas las firmas almacenadas en la base de datos."""
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()
    cursor.execute("SELECT firma FROM firmas")
    firmas = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return firmas

@app.route('/comparar', methods=['POST'])
def comparar_firma():
    # Espera un JSON con {"foto": "<base64>"}
    data = request.get_json()
    if not data or 'foto' not in data:
        return jsonify({"error": "Falta la foto en base64"}), 400

    # Guarda la imagen temporalmente
    foto_b64 = data['foto']
    try:
        img_bytes = base64.b64decode(foto_b64)
    except Exception:
        return jsonify({"error": "Base64 inválido"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(img_bytes)
        tmp_path = tmp.name

    # Firma la imagen recibida
    try:
        firma_nueva = sign_image(tmp_path, priv_key)
    finally:
        os.remove(tmp_path)

    # Obtiene firmas de la base de datos
    firmas_db = obtener_firmas_db()
    if not firmas_db:
        return jsonify({"error": "No hay firmas en la base de datos"}), 404

    # Compara la firma nueva con las de la base de datos
    max_sim = -1
    firma_mas_similar = None
    for firma_db in firmas_db:
        try:
            sim = comparator.compare(firma_nueva, firma_db)
            if sim > max_sim:
                max_sim = sim
                firma_mas_similar = firma_db
        except Exception:
            continue  # Si alguna comparación falla, la ignora

    if firma_mas_similar is None:
        return jsonify({"error": "No se pudo comparar firmas"}), 500

    return jsonify({
        "firma_mas_similar": firma_mas_similar,
        "similitud": round(max_sim * 100, 2)
    })

if __name__ == "__main__":
    app.run(host=HOST, port=PORT)
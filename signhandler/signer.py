import cv2
import hashlib
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

def capture_square_photo(filename='photo.jpg', size=256):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("No se pudo abrir la cámara.")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise Exception("No se pudo capturar la imagen.")
    h, w, _ = frame.shape
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    square = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
    square = cv2.resize(square, (size, size))
    cv2.imwrite(filename, square)
    return filename

def generate_keys():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()
    return private_key, public_key

def sign_image(image_path, private_key):
    with open(image_path, 'rb') as f:
        data = f.read()
    digest = hashlib.sha256(data).digest()
    signature = private_key.sign(
        digest,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    return base64.b64encode(signature).decode('utf-8')

def save_public_key(public_key, filename='public_key.pem'):
    pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    with open(filename, 'wb') as f:
        f.write(pem)

if __name__ == "__main__":
    photo_path = capture_square_photo()
    priv_key, pub_key = generate_keys()
    signature = sign_image(photo_path, priv_key)
    save_public_key(pub_key)
    print(f"Foto guardada en: {photo_path}")
    print(f"Firma digital (base64): {signature}")
    print("Clave pública guardada en: public_key.pem")
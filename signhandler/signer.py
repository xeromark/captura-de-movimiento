import cv2
import torch
import numpy as np
import base64
import json
from torchvision import transforms
from PIL import Image
from signhandler.siamese_network import SiameseNetwork

class FaceEmbeddingGenerator:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        
        # Crear instancia del modelo y cargar pesos desde el modelo arreglado
        self.model = SiameseNetwork(embedding_size=128)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Transformaciones específicas según el código de Discord
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5553568005561829, 0.39241111278533936, 0.3086508810520172], 
                std=[0.19446837902069092, 0.16089946031570435, 0.1428135633468628]
            )
        ])
    
    def preprocess_image(self, image):
        """
        Preprocesa la imagen para el modelo usando las transformaciones específicas.
        Entrada: imagen numpy (BGR de OpenCV)
        Salida: tensor [1, 3, 224, 224] para el modelo
        """
        # Convertir BGR a RGB
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convertir numpy array a PIL Image
        pil_image = Image.fromarray(image)
        
        # Aplicar transformaciones específicas
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        return image_tensor
    
    def generate_embedding(self, image):
        """
        Genera embedding facial a partir de una imagen.
        Entrada: imagen numpy (H, W, C) en formato BGR de OpenCV
        Salida: embedding como string base64 para almacenar en DB
        """
        try:
            print(f"🔍 DEBUG - Imagen entrada: {image.shape}")
            
            with torch.no_grad():
                # Preprocesar imagen
                image_tensor = self.preprocess_image(image)
                print(f"🔍 DEBUG - Tensor procesado: {image_tensor.shape}")
                
                # Generar embedding con el modelo
                embedding = self.model(image_tensor)
                print(f"🔍 DEBUG - Embedding generado: {embedding.shape}")
                
                # Convertir a numpy y luego a lista para serialización
                embedding_np = embedding.cpu().numpy().flatten()
                embedding_list = embedding_np.tolist()
                
                # Serializar a JSON y luego a base64 para almacenamiento
                embedding_json = json.dumps(embedding_list)
                embedding_b64 = base64.b64encode(embedding_json.encode('utf-8')).decode('utf-8')
                
                print(f"✅ Embedding serializado correctamente")
                return embedding_b64
                
        except Exception as e:
            print(f"❌ Error generando embedding: {e}")
            print(f"   Imagen shape: {image.shape if hasattr(image, 'shape') else 'N/A'}")
            import traceback
            traceback.print_exc()
            return None

def capture_square_photo(filename='photo.jpg', size=256):
    """Captura una foto cuadrada de la cámara"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("No se pudo abrir la cámara.")
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise Exception("No se pudo capturar la imagen.")
    
    # Hacer la imagen cuadrada
    h, w, _ = frame.shape
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    square = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
    
    # Redimensionar al tamaño deseado
    square = cv2.resize(square, (size, size))
    cv2.imwrite(filename, square)
    
    return filename, square

def generate_face_signature(image_or_path, model_path='signhandler/model.pth', device='cpu'):
    """
    Genera la firma facial (embedding) a partir de una imagen o ruta.
    Entrada: imagen numpy o ruta del archivo
    Salida: embedding en formato base64 para la base de datos
    """
    generator = FaceEmbeddingGenerator(model_path, device)
    
    if isinstance(image_or_path, str):
        # Es una ruta de archivo
        image = cv2.imread(image_or_path)
        if image is None:
            raise Exception(f"No se pudo cargar la imagen: {image_or_path}")
    else:
        # Es una imagen numpy
        image = image_or_path
    
    return generator.generate_embedding(image)

if __name__ == "__main__":
    try:
        # Capturar foto
        photo_path, photo_array = capture_square_photo()
        print(f"✅ Foto guardada en: {photo_path}")
        
        # Generar embedding facial
        embedding = generate_face_signature(photo_array)
        print(f"✅ Embedding facial generado (primeros 100 chars): {embedding[:100]}...")
        print(f"📏 Tamaño del embedding: {len(embedding)} caracteres")
        
    except Exception as e:
        print(f"❌ Error: {e}")
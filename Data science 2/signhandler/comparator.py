import torch
import torch.nn.functional as F
import numpy as np
import base64
import json
import cv2
from signhandler.siamese_network import SiameseNetwork

class SignatureComparator:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        
        # Truco para cargar modelo con referencia __main__.SiameseNetwork
        import __main__
        __main__.SiameseNetwork = SiameseNetwork
        
        try:
            # Cargar el modelo original
            self.model = torch.load(model_path, map_location=self.device, weights_only=False)
        finally:
            # Limpiar la referencia temporal
            if hasattr(__main__, 'SiameseNetwork'):
                delattr(__main__, 'SiameseNetwork')
        
        self.model.to(self.device)
        self.model.eval()

    def embedding_from_base64(self, embedding_b64):
        """
        Convierte un embedding en base64 de vuelta a tensor de PyTorch.
        Entrada: string base64 que contiene el embedding serializado
        Salida: tensor de PyTorch
        """
        try:
            # Decodificar de base64 a JSON
            embedding_json = base64.b64decode(embedding_b64).decode('utf-8')
            
            # Deserializar de JSON a lista
            embedding_list = json.loads(embedding_json)
            
            # Convertir a tensor de PyTorch
            embedding_tensor = torch.tensor(embedding_list, device=self.device, dtype=torch.float32)
            
            return embedding_tensor
            
        except Exception as e:
            print(f"⚠️ Error al decodificar embedding: {e}")
            return None

    def preprocess_image_to_embedding(self, image):
        """
        Convierte una imagen directamente en embedding usando el modelo.
        Entrada: imagen numpy (H, W, C) en formato BGR de OpenCV
        Salida: tensor embedding
        """
        try:
            # Convertir BGR a RGB
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Redimensionar a 256x256
            image = cv2.resize(image, (256, 256))
            
            # Normalizar a [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Convertir a tensor y reordenar dimensiones: [H, W, C] -> [C, H, W]
            image_tensor = torch.from_numpy(image).permute(2, 0, 1)
            
            # Añadir dimensión de batch: [C, H, W] -> [1, C, H, W]
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # Generar embedding con el modelo
            with torch.no_grad():
                embedding = self.model(image_tensor)
                return embedding.squeeze(0)  # Remover dimensión de batch
                
        except Exception as e:
            print(f"⚠️ Error al procesar imagen a embedding: {e}")
            return None

    def get_embedding(self, signature_or_image):
        """
        Obtiene el embedding de una firma (base64) o imagen.
        
        Args:
            signature_or_image: Puede ser:
                - String base64 con embedding pre-calculado
                - Imagen numpy para generar embedding en tiempo real
        
        Returns:
            tensor: Embedding normalizado
        """
        if isinstance(signature_or_image, str):
            # Es un embedding en base64 pre-calculado
            embedding = self.embedding_from_base64(signature_or_image)
            if embedding is not None:
                # Normalizar L2
                embedding = F.normalize(embedding.unsqueeze(0), p=2, dim=1)
                return embedding
        else:
            # Es una imagen numpy, generar embedding
            embedding = self.preprocess_image_to_embedding(signature_or_image)
            if embedding is not None:
                # Normalizar L2
                embedding = F.normalize(embedding.unsqueeze(0), p=2, dim=1)
                return embedding
        
        # Si falló todo, retornar None
        return None

    def compare(self, signature1, signature2):
        """
        Compara dos firmas/imágenes usando distancia euclidiana de sus embeddings.
        
        Args:
            signature1: Primera firma (base64) o imagen (numpy)
            signature2: Segunda firma (base64) o imagen (numpy)
            
        Returns:
            float: Distancia euclidiana (menor valor = más similar)
        """
        try:
            e1 = self.get_embedding(signature1)
            e2 = self.get_embedding(signature2)
            
            if e1 is None or e2 is None:
                print("⚠️ Error: No se pudo obtener embedding para una de las firmas")
                return float('inf')  # Distancia infinita si hay error
            
            # Distancia euclidiana usando torch.nn.functional.pairwise_distance
            distance = F.pairwise_distance(e1, e2).item()
            return distance
            
        except Exception as e:
            print(f"⚠️ Error en comparación: {e}")
            return float('inf')
    
    def is_known(self, signature1, signature2, threshold=1.0):
        """
        Determina si dos firmas/imágenes corresponden a la misma persona.
        
        Args:
            signature1: Primera firma (base64) o imagen (numpy)
            signature2: Segunda firma (base64) o imagen (numpy)
            threshold: Umbral de distancia (default: 1.0). Menor valor = más estricto
            
        Returns:
            bool: True si la distancia es menor al threshold (conocido), False si no
        """
        distance = self.compare(signature1, signature2)
        return distance < threshold

    def distance_to_similarity_percentage(self, distance, max_distance=2.0):
        """
        Convierte distancia euclidiana a porcentaje de similitud.
        
        Args:
            distance: Distancia euclidiana
            max_distance: Distancia máxima esperada (default: 2.0 para embeddings normalizados)
            
        Returns:
            float: Porcentaje de similitud (0-100)
        """
        # Convertir distancia a similitud usando función exponencial
        # Distancia 0 = 100% similitud, distancia máxima = 0% similitud
        if distance >= max_distance:
            return 0.0
        
        # Función exponencial decreciente
        similarity = np.exp(-distance) * 100
        return min(100.0, max(0.0, similarity))

    def compare_with_all(self, query_signature_or_image, db_signatures, threshold=1.0, debug=True):
        """
        Compara una firma/imagen query con todas las firmas en la base de datos.
        
        Args:
            query_signature_or_image: Firma query (base64) o imagen (numpy)
            db_signatures: Lista de firmas de la DB [(id, firma_base64), ...]
            threshold: Umbral de distancia para clasificación
            debug: Si mostrar información de debug
            
        Returns:
            tuple: (mejor_match_id, distancia, porcentaje_similitud, es_conocido)
        """
        best_match_id = None
        best_distance = float('inf')
        best_similarity = 0.0
        
        if debug:
            print(f"🔍 Comparando con {len(db_signatures)} firmas en la DB...")
        
        for db_id, db_signature in db_signatures:
            try:
                distance = self.compare(query_signature_or_image, db_signature)
                similarity = self.distance_to_similarity_percentage(distance)
                
                if debug:
                    print(f"  ID {db_id}: distancia={distance:.4f}, similitud={similarity:.2f}%")
                
                if distance < best_distance:
                    best_distance = distance
                    best_match_id = db_id
                    best_similarity = similarity
                    
            except Exception as e:
                if debug:
                    print(f"  ⚠️ Error comparando con ID {db_id}: {e}")
                continue
        
        is_known = best_distance < threshold
        
        if debug:
            status = "CONOCIDO" if is_known else "DESCONOCIDO"
            print(f"🎯 Mejor match: ID {best_match_id}, distancia={best_distance:.4f}, similitud={best_similarity:.2f}% -> {status}")
        
        return best_match_id, best_distance, best_similarity, is_known

    def compare_with_discord_logic(self, signature1, signature2):
        """
        Compara dos firmas usando la lógica exacta del código de Discord.
        
        Este método implementa exactamente:
        distance = torch.nn.functional.pairwise_distance(output1, output2).item()
        
        Args:
            signature1: Primera firma (embedding) 
            signature2: Segunda firma (embedding)
            
        Returns:
            float: Distancia euclidiana entre los embeddings
        """
        # Obtener embeddings (output1, output2 en Discord)
        output1 = self.get_embedding(signature1)
        output2 = self.get_embedding(signature2)
        
        if output1 is None or output2 is None:
            return float('inf')
        
        # Comparación exacta según Discord:
        # distance = torch.nn.functional.pairwise_distance(output1, output2).item()
        distance = F.pairwise_distance(output1, output2).item()
        
        return distance
    
    def is_known_discord_logic(self, signature1, signature2, threshold=2.5):
        """
        Determina si dos firmas corresponden a la misma persona usando lógica de Discord.
        
        Este método implementa exactamente:
        if (distances < threshold): conocido = true
        
        Args:
            signature1: Primera firma
            signature2: Segunda firma  
            threshold: Umbral de distancia (default: 2.5)
            
        Returns:
            tuple: (conocido: bool, distance: float)
        """
        distance = self.compare_with_discord_logic(signature1, signature2)
        
        # Definición de umbral exacta según Discord:
        # if (distances < threshold): conocido = true
        conocido = distance < threshold
        
        return conocido, distance

if __name__ == "__main__":
    # Ejemplo de uso:
    # Carga tus firmas como listas o arrays
    firma1_array = [0.1, 0.2, 0.3, 0.4]  # Reemplaza con tu firma real
    firma2_array = [0.1, 0.2, 0.3, 0.5]  # Reemplaza con tu firma real

    # Inicializa el comparador con la ruta a tu modelo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    comparator = SignatureComparator('ruta/al/modelo.pt', device=device)
    
    # Comparar usando distancia euclidiana
    distance = comparator.compare(firma1_array, firma2_array)
    print(f"Distancia euclidiana: {distance:.4f}")
    
    # Determinar si es conocido o desconocido
    threshold = 1.0  # Ajustar según tus necesidades
    is_same_person = comparator.is_known(firma1_array, firma2_array, threshold)
    print(f"¿Es la misma persona? {'Sí' if is_same_person else 'No'}")

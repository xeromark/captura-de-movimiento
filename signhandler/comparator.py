import torch
import torch.nn.functional as F
from signhandler.siamese_network import SiameseNetwork

class SignatureComparator:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        # Carga tu modelo entrenado (state_dict o modelo completo)
        # Usamos weights_only=False para permitir cargar clases personalizadas
        self.model = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.eval()

    def preprocess(self, signature):
        """
        Convierte la firma (lista/np.array) en un tensor normalizado
        Ajusta normalización/reshape según tu caso.
        """
        x = torch.tensor(signature, dtype=torch.float32)
        # ejemplo: normalizar al rango [0,1] o (x - μ) / σ
        # x = (x - x.mean()) / (x.std() + 1e-6)
        return x.to(self.device).unsqueeze(0)  # shape: [1, features]

    def get_embedding(self, signature):
        """
        Pasa la firma por el modelo para obtener un embedding L2-normalizado.
        """
        x = self.preprocess(signature)
        with torch.no_grad():
            emb = self.model(x)            # shape: [1, emb_dim]
            emb = F.normalize(emb, p=2, dim=1)
        return emb

    def compare(self, signature1, signature2):
        """
        Compara dos firmas por similitud coseno de sus embeddings.
        """
        e1 = self.get_embedding(signature1)
        e2 = self.get_embedding(signature2)
        # Cosine similarity: dot product entre vectores unitarios
        sim = torch.sum(e1 * e2, dim=1).item()
        return sim

if __name__ == "__main__":
    # Ejemplo de uso:
    # Carga tus firmas como listas o arrays
    firma1_array = [0.1, 0.2, 0.3, 0.4]  # Reemplaza con tu firma real
    firma2_array = [0.1, 0.2, 0.3, 0.5]  # Reemplaza con tu firma real

    # Inicializa el comparador con la ruta a tu modelo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    comparator = SignatureComparator('ruta/al/modelo.pt', device=device)

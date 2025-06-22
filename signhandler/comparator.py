import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

class SignatureComparator:
    def __init__(self, model_path):
        # Carga el modelo previamente entrenado
        self.model = joblib.load(model_path)

    def preprocess(self, signature):
        # Preprocesa la firma (ajusta según tu caso)
        # Por ejemplo: normalización, reshape, etc.
        return np.array(signature).reshape(1, -1)

    def get_features(self, signature):
        # Extrae características usando el modelo
        processed = self.preprocess(signature)
        features = self.model.transform(processed)
        return features

    def compare(self, signature1, signature2):
        # Extrae características de ambas firmas
        feat1 = self.get_features(signature1)
        feat2 = self.get_features(signature2)
        # Calcula la similitud coseno
        similarity = cosine_similarity(feat1, feat2)[0][0]
        return similarity

# Ejemplo de uso:
# comparator = SignatureComparator('ruta/al/modelo.pkl')
# sim = comparator.compare(firma1, firma2)
# print(f"Nivel de similitud: {sim:.2f}")
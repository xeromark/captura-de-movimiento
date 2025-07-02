#!/usr/bin/env python3
"""
Script para crear un modelo de prueba con la arquitectura correcta.
Este modelo no estará entrenado, pero tendrá la estructura correcta para hacer pruebas.
"""

import torch
import os
from signhandler.siamese_network import SiameseNetwork

def create_test_model():
    """Crea un modelo de prueba con pesos aleatorios."""
    
    # Crear modelo con la arquitectura correcta
    model = SiameseNetwork(embedding_size=128)
    
    # El modelo ya tiene pesos aleatorios por defecto
    print(f"✅ Modelo creado con arquitectura:")
    print(f"   - Entrada: imágenes 256x256x3")
    print(f"   - Salida: embeddings de 128 dimensiones")
    
    # Crear directorio si no existe
    os.makedirs("signhandler", exist_ok=True)
    
    # Guardar modelo
    model_path = "signhandler/model.pth"
    torch.save(model, model_path)
    print(f"✅ Modelo guardado en: {model_path}")
    
    # Probar que el modelo funciona
    test_input = torch.randn(1, 3, 256, 256)  # Imagen de prueba
    with torch.no_grad():
        output = model(test_input)
        print(f"✅ Prueba exitosa - Entrada: {test_input.shape}, Salida: {output.shape}")
    
    return model_path

if __name__ == "__main__":
    create_test_model()

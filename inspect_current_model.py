#!/usr/bin/env python3
"""
Script para inspeccionar el modelo existente y ver su arquitectura.
"""

import torch
import sys
import os

# Añadir el directorio al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "signhandler"))

def inspect_model():
    model_path = "signhandler/model.pth"
    
    try:
        print(f"🔍 Inspeccionando modelo: {model_path}")
        
        # Cargar modelo
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        
        print(f"✅ Modelo cargado exitosamente")
        print(f"📋 Tipo: {type(model)}")
        print(f"📋 Clase: {model.__class__.__name__}")
        
        print("\n🏗️ Arquitectura del modelo:")
        print(model)
        
        print("\n📊 Parámetros del modelo:")
        total_params = 0
        for name, param in model.named_parameters():
            print(f"  {name}: {param.shape} ({param.numel()} parámetros)")
            total_params += param.numel()
        
        print(f"\n📈 Total de parámetros: {total_params:,}")
        
        # Probar con una entrada de ejemplo
        print("\n🧪 Probando con entrada de ejemplo...")
        try:
            # Probar con imagen 256x256x3
            test_input = torch.randn(1, 3, 256, 256)
            print(f"   Entrada: {test_input.shape}")
            
            with torch.no_grad():
                output = model(test_input)
                print(f"   ✅ Salida: {output.shape}")
                print(f"   ✅ El modelo funciona correctamente con imágenes 256x256x3")
        except Exception as e:
            print(f"   ❌ Error con entrada 256x256x3: {e}")
            
            # Probar con otras dimensiones
            test_sizes = [
                (1, 128),      # Vector 1D
                (1, 3, 224, 224),  # Imagen 224x224
                (1, 784),      # Vector aplanado (28x28)
            ]
            
            for test_shape in test_sizes:
                try:
                    test_input = torch.randn(*test_shape)
                    print(f"   Probando con {test_shape}...")
                    with torch.no_grad():
                        output = model(test_input)
                        print(f"   ✅ Funciona con {test_shape} -> {output.shape}")
                        break
                except Exception as e2:
                    print(f"   ❌ No funciona con {test_shape}: {e2}")
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")

if __name__ == "__main__":
    inspect_model()

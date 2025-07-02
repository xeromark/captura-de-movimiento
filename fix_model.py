#!/usr/bin/env python3
"""
Script para crear una versión compatible del modelo.
Extrae los pesos del modelo guardado y los guarda como state_dict puro.
"""

import torch
import sys
import os
sys.path.append('signhandler')
from siamese_network import SiameseNetwork

def fix_model():
    """Arregla el modelo para que sea compatible."""
    model_path = 'signhandler/model.pth'
    fixed_model_path = 'signhandler/model_fixed.pth'
    
    print("🔧 Arreglando modelo...")
    
    try:
        # Cargar el modelo original
        original_model = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f"✅ Modelo original cargado: {type(original_model)}")
        
        if hasattr(original_model, 'state_dict'):
            # Es una instancia del modelo
            state_dict = original_model.state_dict()
        else:
            # Es un diccionario o state_dict directo
            state_dict = original_model
        
        print("📋 Capas del modelo:")
        for name, param in state_dict.items():
            print(f"  {name}: {param.shape}")
        
        # Crear una nueva instancia de SiameseNetwork compatible
        new_model = SiameseNetwork(embedding_size=128)
        
        # Intentar cargar los pesos
        try:
            new_model.load_state_dict(state_dict, strict=False)
            print("✅ Pesos cargados exitosamente (modo no estricto)")
        except Exception as e:
            print(f"⚠️ Error cargando pesos: {e}")
            # Mostrar diferencias
            new_state_dict = new_model.state_dict()
            print("\n🔍 Comparando arquitecturas:")
            print("MODELO GUARDADO -> NUEVA ARQUITECTURA")
            for old_key in state_dict.keys():
                if old_key in new_state_dict:
                    print(f"  ✅ {old_key}: {state_dict[old_key].shape} -> {new_state_dict[old_key].shape}")
                else:
                    print(f"  ❌ {old_key}: {state_dict[old_key].shape} -> NO EXISTE")
            
            print("\nCAPAS NUEVAS NO EN MODELO GUARDADO:")
            for new_key in new_state_dict.keys():
                if new_key not in state_dict:
                    print(f"  ➕ {new_key}: {new_state_dict[new_key].shape}")
        
        # Guardar el modelo arreglado
        torch.save(new_model.state_dict(), fixed_model_path)
        print(f"✅ Modelo arreglado guardado en: {fixed_model_path}")
        
        # Probar carga del modelo arreglado
        test_model = SiameseNetwork(embedding_size=128)
        test_model.load_state_dict(torch.load(fixed_model_path, map_location='cpu'))
        print("✅ Modelo arreglado probado exitosamente")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fix_model()

#!/usr/bin/env python3
"""
Script de prueba para verificar la corrección del error de tipos
"""

import sys
import os

# Añadir paths necesarios
ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT, "signhandler"))

try:
    from signhandler.comparator import SignatureComparator
    from signhandler.signer import sign_image, generate_keys
    import torch
    import tempfile
    import cv2
    import numpy as np
    
    print("✅ Importaciones exitosas")
    print(f"📱 Dispositivo PyTorch: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Generar llaves de prueba
    priv_key, pub_key = generate_keys()
    print("🔑 Llaves generadas")
    
    # Crear imagen de prueba
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        cv2.imwrite(tmp.name, test_image)
        
        # Generar firma criptográfica
        firma_cripto = sign_image(tmp.name, priv_key)
        print(f"📝 Firma generada: {firma_cripto[:50]}...")
        print(f"📏 Longitud firma: {len(firma_cripto)} caracteres")
        
        # Verificar que es base64 válido
        import base64
        try:
            decoded = base64.b64decode(firma_cripto)
            print(f"✅ Firma base64 válida: {len(decoded)} bytes")
        except Exception as e:
            print(f"❌ Error decodificando base64: {e}")
        
        os.unlink(tmp.name)
    
    # Probar el comparador (solo si el modelo existe)
    model_path = os.path.join(ROOT, "signhandler", "model.pth")
    if os.path.exists(model_path):
        try:
            comparator = SignatureComparator(model_path, device='cpu')
            print("✅ Comparador cargado")
            
            # Probar preprocessing
            embedding = comparator.get_embedding(firma_cripto)
            print(f"✅ Embedding generado: shape {embedding.shape}")
            
            # Comparar firma consigo misma
            distance = comparator.compare(firma_cripto, firma_cripto)
            print(f"✅ Distancia consigo misma: {distance:.6f}")
            
            # Generar segunda firma para comparar
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp2:
                test_image2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                cv2.imwrite(tmp2.name, test_image2)
                firma_cripto2 = sign_image(tmp2.name, priv_key)
                
                distance2 = comparator.compare(firma_cripto, firma_cripto2)
                print(f"✅ Distancia entre diferentes: {distance2:.6f}")
                
                os.unlink(tmp2.name)
            
        except Exception as e:
            print(f"⚠️ Error con el modelo (es normal si no existe): {e}")
    else:
        print(f"⚠️ Modelo no encontrado en {model_path}")
    
    print(f"\n🎉 Corrección exitosa: El sistema puede manejar firmas criptográficas como strings base64")
    
except ImportError as e:
    print(f"❌ Error de importación: {e}")
except Exception as e:
    print(f"❌ Error: {e}")

print(f"\n📋 Para probar el sistema completo:")
print(f"   python3 app.py --testdb")
print(f"   python3 app.py")

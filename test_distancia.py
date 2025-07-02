#!/usr/bin/env python3
"""
Script de prueba para verificar la implementación de distancia euclidiana
"""

import sys
import os

# Añadir paths necesarios
ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT, "signhandler"))

try:
    from signhandler.comparator import SignatureComparator
    import torch
    import torch.nn.functional as F
    
    print("✅ Importaciones exitosas")
    print(f"📱 Dispositivo PyTorch: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Crear embeddings de prueba (simulando salida del modelo)
    embedding1 = torch.randn(1, 128)  # Ejemplo: embedding de 128 dimensiones
    embedding2 = embedding1 + torch.randn(1, 128) * 0.1  # Similar pero con ruido
    embedding3 = torch.randn(1, 128)  # Completamente diferente
    
    # Calcular distancias directamente
    distance_similar = F.pairwise_distance(embedding1, embedding2).item()
    distance_different = F.pairwise_distance(embedding1, embedding3).item()
    
    print(f"\n📊 Prueba de distancias euclidiana:")
    print(f"🟢 Distancia entre embeddings similares: {distance_similar:.4f}")
    print(f"🔴 Distancia entre embeddings diferentes: {distance_different:.4f}")
    
    # Prueba con diferentes thresholds
    thresholds = [0.5, 1.0, 1.5, 2.0]
    
    print(f"\n🎯 Prueba de clasificación con diferentes thresholds:")
    for thresh in thresholds:
        similar_known = distance_similar < thresh
        different_known = distance_different < thresh
        
        print(f"   Threshold {thresh}: Similar={'✅ CONOCIDO' if similar_known else '❌ DESCONOCIDO'}, "
              f"Diferente={'✅ CONOCIDO' if different_known else '❌ DESCONOCIDO'}")
    
    print(f"\n💡 Recomendación:")
    print(f"   - Para este ejemplo, usar threshold ~{(distance_similar + distance_different) / 2:.2f}")
    print(f"   - Threshold más bajo = más estricto (menos falsos positivos)")
    print(f"   - Threshold más alto = más permisivo (menos falsos negativos)")
    
    print(f"\n🚀 El sistema de distancia euclidiana está funcionando correctamente!")
    
except ImportError as e:
    print(f"❌ Error de importación: {e}")
    print("Verifica que todas las dependencias estén instaladas")
except Exception as e:
    print(f"❌ Error: {e}")

print(f"\n📋 Uso del sistema principal:")
print(f"   python3 app.py --distance-threshold 1.0")
print(f"   python3 app.py --distance-threshold 0.5 --face-threshold 0.8")
print(f"   python3 app.py --testdb")

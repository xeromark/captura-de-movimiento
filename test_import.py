#!/usr/bin/env python3
# Test script to verify model loading

import os
import sys
import traceback

# Add paths
ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT, "signhandler"))

# Import SiameseNetwork to make it available in __main__ context
from siamese_network import SiameseNetwork

try:
    print("🔍 Probando importación de torch...")
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
    
    print("🔍 Probando importación de comparator...")
    from comparator import SignatureComparator
    print("✅ SignatureComparator importado correctamente")
    
    print("🔍 Probando inicialización del comparator...")
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "signhandler", "model.pth")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📍 Modelo en: {MODEL_PATH}")
    print(f"🖥️  Dispositivo: {DEVICE}")
    
    if os.path.exists(MODEL_PATH):
        print("✅ Archivo del modelo existe")
        comparator = SignatureComparator(MODEL_PATH, device=DEVICE)
        print("✅ SignatureComparator inicializado correctamente")
    else:
        print("❌ Archivo del modelo no encontrado")
        
    print("🔍 Probando importación completa de signhandler.app...")
    import signhandler.app
    print("✅ signhandler.app importado correctamente")
    
    print("\n🎉 ¡Todas las pruebas pasaron exitosamente!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\n📋 Traceback completo:")
    traceback.print_exc()

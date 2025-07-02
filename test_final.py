#!/usr/bin/env python3
# Test final para verificar que el problema se solucionó

import sys
import traceback

try:
    # Probar la importación que estaba fallando
    from signhandler.app import app as signhandler_app
    print("✅ Importación exitosa de signhandler.app")
    print("✅ El problema del modelo se ha solucionado correctamente")
    
    # Probar que la aplicación puede inicializarse
    print("🔍 Verificando configuración de la aplicación...")
    print(f"📍 Aplicación Flask configurada: {signhandler_app.name}")
    print("✅ La aplicación está lista para ejecutarse")
    
except Exception as e:
    print(f"❌ Error durante la importación: {e}")
    print("\n📋 Traceback completo:")
    traceback.print_exc()
    sys.exit(1)

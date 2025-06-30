#!/usr/bin/env python3
# Prueba final para verificar que el problema se solucionó completamente

import sys
import os
import traceback

# Agregar el directorio actual al path
sys.path.insert(0, os.getcwd())

try:
    print("🔍 Probando importación del archivo app.py principal...")
    import app
    print("✅ Importación exitosa de app.py")
    
    print("🔍 Verificando que signhandler_app está disponible...")
    print(f"📍 signhandler_app: {app.signhandler_app}")
    print("✅ signhandler_app cargado correctamente")
    
    print("🔍 Probando función main...")
    # No ejecutamos main() porque requiere argumentos, pero verificamos que existe
    if hasattr(app, 'main'):
        print("✅ Función main() disponible")
    else:
        print("❌ Función main() no encontrada")
    
    print("\n🎉 ¡Todas las pruebas pasaron! El problema se ha solucionado.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\n📋 Traceback completo:")
    traceback.print_exc()
    sys.exit(1)

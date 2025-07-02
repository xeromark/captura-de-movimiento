#!/usr/bin/env python3
"""
Script simple para probar la base de datos sin cargar el modelo PyTorch
"""

import os
import psycopg2
from dotenv import load_dotenv
from urllib.parse import urlparse

# Cargar variables de entorno
load_dotenv()

def get_db_params():
    """Obtiene parámetros de BD desde variables de entorno"""
    database_url = os.getenv('DATABASE_URL')
    
    if database_url:
        parsed = urlparse(database_url)
        return {
            'dbname': parsed.path[1:],
            'user': parsed.username,
            'password': parsed.password,
            'host': parsed.hostname,
            'port': parsed.port or 5432
        }

def probar_conexion():
    """Prueba la conexión a la base de datos"""
    try:
        db_params = get_db_params()
        print(f"📋 Conectando a: {db_params}")
        
        with psycopg2.connect(**db_params) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()[0]
                print(f"✅ Conexión BD exitosa: {version}")
                
                # Crear tabla
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS firmas (
                        id SERIAL PRIMARY KEY,
                        firma TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
                print("✅ Tabla 'firmas' verificada/creada")
                
                return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Probando conexión a la base de datos...")
    if probar_conexion():
        print("✅ Base de datos lista para usar")
    else:
        print("❌ Problemas con la base de datos")

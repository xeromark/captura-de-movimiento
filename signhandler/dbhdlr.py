import psycopg2

# Parámetros de conexión a la base de datos
db_params = {
    'dbname': 'tu_basededatos',
    'user': 'tu_usuario',
    'password': 'tu_contraseña',
    'host': 'localhost',
    'port': 5432
}


def insertar_firma(db_params, firma):
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS firmas (
            id SERIAL PRIMARY KEY,
            firma TEXT NOT NULL
        )
    """)
    cursor.execute("INSERT INTO firmas (firma) VALUES (%s)", (firma,))
    conn.commit()
    cursor.close()
    conn.close()

# Ejemplo de uso:
# db_params = {
#     'dbname': 'tu_basededatos',
#     'user': 'tu_usuario',
#     'password': 'tu_contraseña',
#     'host': 'localhost',
#     'port': 5432
# }
# insertar_firma(db_params, 'mi_firma_digital')
#!/bin/bash
# Script para inicializar la base de datos PostgreSQL

echo "🔧 Inicializando base de datos PostgreSQL..."

# Crear la base de datos si no existe
echo "📊 Creando base de datos 'signatures'..."
psql -U postgres -h localhost -c "CREATE DATABASE signatures;" 2>/dev/null || echo "Base de datos ya existe"

# Conectar a la base de datos y crear la tabla
echo "🗃️ Creando tabla 'firmas'..."
psql -U postgres -h localhost -d signatures << EOF
CREATE TABLE IF NOT EXISTS firmas (
    id SERIAL PRIMARY KEY,
    firma TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Verificar que la tabla se creó
\dt

-- Mostrar estructura de la tabla
\d firmas

-- Limpiar datos de prueba anteriores si existen
DELETE FROM firmas WHERE firma LIKE 'firma_prueba_%';

-- Verificar que la tabla está lista para embeddings reales
SELECT COUNT(*) as total_firmas FROM firmas;

EOF

echo "✅ Inicialización completada"

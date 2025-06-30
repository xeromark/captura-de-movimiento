# Sistema Integrado de Captura y Procesamiento de Firmas

## Descripción
Archivo ejecutable único que combina todas las funcionalidades de captura de cámara y procesamiento de firmas.

## Funcionalidades Integradas

### 🔑 Procesamiento de Firmas
- Generación automática de firmas criptográficas para imágenes
- Comparación de firmas usando red neuronal SiameseNetwork
- Almacenamiento en base de datos PostgreSQL
- API REST para comparación de firmas

### 📷 Captura de Imágenes  
- Captura desde cámaras IP (múltiples formatos de stream)
- Captura desde webcam local
- Detección automática de caras
- Escaneo de red para encontrar cámaras

### 🌐 Servidor Web
- API REST con Flask
- Endpoint `/comparar` para análisis de firmas
- Endpoint `/upload` para recibir imágenes
- Procesamiento automático de imágenes recibidas

## Comandos Disponibles

```bash
# Iniciar servidor REST
python app.py server --host 0.0.0.0 --port 5000

# Capturar desde cámara IP
python app.py camera 192.168.1.50 --dest-ip 192.168.1.100

# Ejecutar servidor + captura simultáneamente
python app.py full 192.168.1.50 --host 0.0.0.0 --port 5000

# Capturar foto manual desde webcam
python app.py foto

# Escanear red en busca de cámaras
python app.py scan

# Procesar imágenes existentes
python app.py process --carpeta capturas

# Enviar imágenes a servidor
python app.py send --dest-ip 192.168.1.100 --dest-port 5000
```

## Arquitectura

El archivo `app.py` consolida:
- **signhandler/**: Procesamiento de firmas y comparación
- **container/**: Captura de cámara y gestión de imágenes  
- **Librerías**: PyTorch, OpenCV, Flask, PostgreSQL

## Flujo de Trabajo

1. **Captura**: Obtiene imágenes desde cámaras IP o webcam
2. **Detección**: Identifica caras en las imágenes
3. **Firma**: Genera firmas criptográficas únicas
4. **Almacenamiento**: Guarda firmas en base de datos
5. **Comparación**: Compara nuevas firmas con existentes
6. **Comunicación**: API REST para integración externa

## Dependencias

- PyTorch (red neuronal)
- OpenCV (procesamiento de imagen)
- Flask (servidor web)
- PostgreSQL (base de datos)
- Cryptography (firmas digitales)

## Ventajas del Enfoque Consolidado

✅ **Un solo ejecutable** - No hay dependencias entre módulos
✅ **Funcionalidad completa** - Combina captura y procesamiento  
✅ **Fácil despliegue** - Un archivo para todas las operaciones
✅ **Múltiples modos** - Servidor, cliente, o ambos
✅ **Extensible** - Fácil agregar nuevas funcionalidades

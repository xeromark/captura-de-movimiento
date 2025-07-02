# Captura de Movimiento

Este repositorio contiene una solución basada en contenedores para la captura, firma digital y comparación de imágenes de movimiento. A continuación se describen los contenedores definidos y los programas principales que ejecutan.

## Contenedores

### 1. `container`
- **Función:** Contenedor base de Python preparado para ejecutar scripts personalizados.
- **Programas principales:** Ejecuta el script `init.sh` al iniciar, permitiendo la personalización del entorno o la ejecución de tareas específicas según el contenido del directorio montado.
- **Descripción:** Este contenedor es genérico y sirve como plantilla para pruebas o desarrollo, ejecutando cualquier script que se coloque en el directorio `container`.

### 2. `signhandler`
- **Función:** Captura, firma digital y comparación de imágenes.
- **Programas principales:** 
    - `app.py`: API REST en Flask para recibir imágenes, firmarlas digitalmente y compararlas con firmas almacenadas.
    - `signer.py`: Captura imágenes, genera claves RSA y firma digitalmente las imágenes.
    - `comparator.py`: Compara firmas digitales usando técnicas de aprendizaje automático.
    - `dbhdlr.py`: Maneja la conexión y operaciones con la base de datos PostgreSQL para almacenar y recuperar firmas.
- **Descripción:** Este contenedor expone una API en el puerto 5000 para recibir imágenes en base64, firmarlas y compararlas con firmas previas. Utiliza aprendizaje automático para la comparación y almacena los resultados en una base de datos.

## Flujo de Trabajo

1. El contenedor `signhandler` recibe una imagen a través de la API.
2. La imagen es firmada digitalmente y comparada con firmas almacenadas en la base de datos.
3. El resultado de la comparación se devuelve a través de la API.
4. El contenedor `container` puede ser usado para tareas auxiliares o personalizadas según sea necesario.

---

Cada contenedor está diseñado para ser independiente y escalable, facilitando el mantenimiento y la ampliación del sistema.

Para correr usar ```compose up```con su sistema de contenedores de preferencia.

# db
```
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/signatures

docker run -d --name postgres-signatures -e POSTGRES_DB=signatures -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:latest

psql -U postgres
```
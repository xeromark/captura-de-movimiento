# Cambios Implementados: Distancia Euclidiana

## Resumen de Cambios

Se ha refactorizado el sistema de comparación de firmas para usar **distancia euclidiana** en lugar de **similitud coseno** como se solicitó.

## Cambios Principales

### 1. Archivo `signhandler/comparator.py`

- **Método `compare()`**: Ahora usa `torch.nn.functional.pairwise_distance()` para calcular la distancia euclidiana
- **Nuevo método `is_known()`**: Determina si dos firmas corresponden a la misma persona basado en un threshold de distancia
- **Lógica**: `distance < threshold` significa "conocido", `distance >= threshold` significa "desconocido"

```python
def compare(self, signature1, signature2):
    """
    Compara dos firmas usando distancia euclidiana de sus embeddings.
    Retorna la distancia (menor valor = más similar).
    """
    e1 = self.get_embedding(signature1)
    e2 = self.get_embedding(signature2)
    # Euclidean distance usando torch.nn.functional.pairwise_distance
    distance = F.pairwise_distance(e1, e2).item()
    return distance

def is_known(self, signature1, signature2, threshold=1.0):
    """
    Determina si dos firmas corresponden a la misma persona.
    """
    distance = self.compare(signature1, signature2)
    return distance < threshold
```

### 2. Archivo `app.py`

#### Nuevos parámetros:
- `--distance-threshold`: Configura el umbral de distancia (default: 1.0)
- `--face-threshold`: Threshold existente para detección de rostros (default: 0.7)

#### Cambios en la clase `IntegratedSystem`:
- **Constructor**: Ahora acepta `distance_threshold` como parámetro
- **Método `comparar_firma_con_db()`**: 
  - Busca la **menor distancia** en lugar de la mayor similitud
  - Retorna la distancia y si es "conocido" o "desconocido"
- **Método `procesar_cara()`**: Retorna la distancia y estado de conocimiento

#### Cambios en la visualización:
- **Rectángulos de color**: Verde para personas conocidas, rojo para desconocidas
- **Texto mostrado**: Muestra "CONOCIDO" o "DESCONOCIDO" junto con la distancia
- **Formato**: `"CONOCIDO (d=0.345)"` o `"DESCONOCIDO (d=1.234)"`

#### Guardado de imágenes:
- Los archivos se guardan con prefijo según su estado: `cara_conocido_timestamp.jpg` o `cara_desconocido_timestamp.jpg`

## Uso del Sistema

### Comandos de ejemplo:

```bash
# Usar threshold de distancia más estricto (0.5)
python3 app.py --distance-threshold 0.5

# Combinar con threshold de detección facial
python3 app.py --face-threshold 0.8 --distance-threshold 0.7

# Usar con cámara IP
python3 app.py --ip 192.168.1.100 --distance-threshold 1.2

# Probar conexión a BD con configuración personalizada
python3 app.py --testdb --distance-threshold 0.8
```

### Interpretación de la distancia:

- **Distancia baja** (< threshold): Personas conocidas (mismo individuo)
- **Distancia alta** (>= threshold): Personas desconocidas (individuos diferentes)
- **Threshold recomendado**: Entre 0.5 (muy estricto) y 2.0 (permisivo)

## Lógica de Decisión

```python
distance = torch.nn.functional.pairwise_distance(output1, output2).item()
if distance < threshold:
    conocido = True  # Persona conocida
else:
    conocido = False  # Persona desconocida
```

## Ventajas de la Distancia Euclidiana

1. **Más intuitiva**: Menor distancia = mayor similitud
2. **Threshold claro**: Un valor fijo que determina el límite de decisión
3. **Escalable**: Fácil de ajustar según la precisión deseada
4. **Consistente**: Resultados más predecibles que la similitud coseno

## Configuración Recomendada

- **Threshold conservador**: 0.5 - Menos falsos positivos, más falsos negativos
- **Threshold balanceado**: 1.0 - Balance entre precisión y recall  
- **Threshold permisivo**: 1.5 - Más detecciones, posibles falsos positivos

## Archivos Modificados

1. `/signhandler/comparator.py` - Lógica de comparación principal
2. `/app.py` - Sistema integrado con nuevos parámetros y visualización
3. `/CAMBIOS_DISTANCIA_EUCLIDANA.md` - Esta documentación

## Estado del Sistema

✅ **Completado**: Refactorización a distancia euclidiana
✅ **Completado**: Configuración de threshold via CLI
✅ **Completado**: Visualización con códigos de color
✅ **Completado**: Clasificación automática conocido/desconocido
✅ **Completado**: Documentación de cambios

El sistema está listo para usar con la nueva lógica de distancia euclidiana.

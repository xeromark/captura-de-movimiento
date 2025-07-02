#!/usr/bin/env python3
"""
Script de prueba para verificar el cálculo de porcentajes de similitud
"""

import math

def calcular_similitud_porcentaje(distancia):
    """
    Convierte distancia euclidiana a porcentaje de similitud
    """
    if distancia == float('inf'):
        return 0.0
    else:
        # Normalizar distancia a un porcentaje (0-100%)
        # Distancia 0 = 100% similitud, distancia alta = 0% similitud
        return max(0, 100 * math.exp(-distancia / 2))

# Pruebas con diferentes distancias
distancias_prueba = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0]

print("🧮 Tabla de Conversión Distancia → Porcentaje de Similitud")
print("=" * 60)
print(f"{'Distancia':<12} {'Similitud %':<12} {'Estado (T=2.5)':<15}")
print("-" * 60)

for dist in distancias_prueba:
    similitud = calcular_similitud_porcentaje(dist)
    estado = "CONOCIDO" if dist < 2.5 else "DESCONOCIDO"
    print(f"{dist:<12.1f} {similitud:<12.1f} {estado:<15}")

print("\n💡 Interpretación:")
print("- Distancia 0.0 = 100% similitud (idénticas)")
print("- Distancia 1.4 ≈ 50% similitud")
print("- Distancia 2.5 ≈ 29% similitud (threshold por defecto)")
print("- Distancia 5.0 ≈ 8% similitud")
print("- Distancia >6.0 ≈ 0% similitud")

print(f"\n🎯 Con threshold actual de 2.5:")
print(f"- Similitud > 29% = CONOCIDO")
print(f"- Similitud ≤ 29% = DESCONOCIDO")

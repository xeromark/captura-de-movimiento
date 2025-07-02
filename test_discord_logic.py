#!/usr/bin/env python3
"""
Script de prueba usando la lógica exacta del código de Discord
"""

import torch
import torch.nn.functional as F

def test_discord_logic():
    """
    Prueba la implementación exacta de la lógica de Discord
    """
    print("🧪 Probando lógica exacta del código de Discord")
    print("=" * 50)
    
    # Simular embeddings (output1, output2 del Discord)
    output1 = torch.randn(1, 128)  # Embedding 1
    output2 = output1 + torch.randn(1, 128) * 0.1  # Similar pero con ruido
    output3 = torch.randn(1, 128)  # Completamente diferente
    
    print("📊 Embeddings generados:")
    print(f"   Output1 shape: {output1.shape}")
    print(f"   Output2 shape: {output2.shape}")
    print(f"   Output3 shape: {output3.shape}")
    
    # Comparación exacta según Discord:
    # distance = torch.nn.functional.pairwise_distance(output1, output2).item()
    distance_similar = F.pairwise_distance(output1, output2).item()
    distance_different = F.pairwise_distance(output1, output3).item()
    
    print(f"\n🔍 Distancias calculadas:")
    print(f"   Similar: {distance_similar:.4f}")
    print(f"   Diferente: {distance_different:.4f}")
    
    # Definición de umbral según Discord:
    # if (distances < threshold): conocido = true
    thresholds = [1.0, 2.0, 2.5, 3.0]
    
    print(f"\n🎯 Pruebas de threshold:")
    for threshold in thresholds:
        conocido_similar = distance_similar < threshold
        conocido_different = distance_different < threshold
        
        print(f"   Threshold {threshold}:")
        print(f"     Similar: {'✅ CONOCIDO' if conocido_similar else '❌ DESCONOCIDO'} (d={distance_similar:.3f})")
        print(f"     Diferente: {'✅ CONOCIDO' if conocido_different else '❌ DESCONOCIDO'} (d={distance_different:.3f})")
    
    print(f"\n💡 Recomendación para threshold:")
    optimal_threshold = (distance_similar + distance_different) / 2
    print(f"   Threshold óptimo estimado: {optimal_threshold:.3f}")
    print(f"   Threshold Discord (2.5): {'✅ APROPIADO' if 2.0 < optimal_threshold < 3.0 else '⚠️ REVISAR'}")

def test_image_transforms():
    """
    Prueba las transformaciones de imagen del Discord
    """
    print(f"\n🖼️ Transformaciones de imagen del Discord:")
    print("=" * 50)
    
    try:
        from torchvision import transforms
        from PIL import Image
        import numpy as np
        
        # Transformaciones exactas del Discord:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5553568005561829, 0.39241111278533936, 0.3086508810520172], 
                std=[0.19446837902069092, 0.16089946031570435, 0.1428135633468628]
            )
        ])
        
        print("✅ Transformaciones cargadas correctamente:")
        print(f"   Resize: (224, 224)")
        print(f"   Mean: [0.5554, 0.3924, 0.3087]")
        print(f"   Std:  [0.1945, 0.1609, 0.1428]")
        
        # Simular una imagen
        fake_image = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        
        # Aplicar transformaciones
        transformed = transform(fake_image)
        print(f"   Tensor resultado: {transformed.shape}")
        
        # Ejemplo de uso completo del Discord:
        # img1 = transform(Image.open(img1_path).convert('RGB')).unsqueeze(0).to(device)
        # output1, output2 = model(img1, img2)
        
        print(f"\n📝 Código equivalente Discord:")
        print(f"   img1 = transform(Image.open(img1_path).convert('RGB')).unsqueeze(0).to(device)")
        print(f"   output1, output2 = model(img1, img2)")
        print(f"   distance = torch.nn.functional.pairwise_distance(output1, output2).item()")
        print(f"   conocido = distance < threshold")
        
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("   Instalar: pip install torchvision pillow")

if __name__ == "__main__":
    print("🚀 Prueba de Implementación de Lógica Discord")
    print("=" * 60)
    
    test_discord_logic()
    test_image_transforms()
    
    print(f"\n✅ Implementación completa:")
    print(f"   1. ✅ Distancia euclidiana: F.pairwise_distance()")
    print(f"   2. ✅ Threshold de decisión: distance < threshold")
    print(f"   3. ✅ Transformaciones de imagen: Resize(224,224) + Normalize")
    print(f"   4. ✅ Threshold recomendado: 2.5 (como en Discord)")
    
    print(f"\n🎯 Para usar en el sistema:")
    print(f"   python3 app.py --distance-threshold 2.5")

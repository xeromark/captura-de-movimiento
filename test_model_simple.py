import torch

# Cargar modelo
try:
    model = torch.load('signhandler/model.pth', map_location='cpu')
    print("Modelo cargado exitosamente")
    print("Tipo:", type(model))
    print("Arquitectura:", model)
    
    # Probar con entrada 256x256x3
    test_input = torch.randn(1, 3, 256, 256)
    output = model(test_input)
    print("Entrada:", test_input.shape)
    print("Salida:", output.shape)
    print("El modelo funciona correctamente!")
    
except Exception as e:
    print("Error:", e)

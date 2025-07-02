#!/usr/bin/env python3
import torch
import sys
from signhandler.siamese_network import SiameseNetwork

try:
    print("Cargando modelo...")
    model = torch.load('signhandler/model.pth', map_location='cpu', weights_only=False)
    
    print("Tipo del modelo:", type(model))
    
    if isinstance(model, dict):
        print("Claves del diccionario:", list(model.keys()))
        if 'state_dict' in model:
            state_dict = model['state_dict']
        else:
            state_dict = model
    else:
        state_dict = model.state_dict() if hasattr(model, 'state_dict') else model
    
    print("\nCapas del modelo:")
    for name, param in state_dict.items():
        print(f"  {name}: {param.shape}")
    
    # Verificar batch norm
    bn_layers = [name for name in state_dict.keys() if 'batch_norm' in name]
    print(f"\nCapas batch_norm: {len(bn_layers)}")
    for layer in bn_layers:
        print(f"  {layer}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

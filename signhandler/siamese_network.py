import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    """
    Red neuronal siamesa para comparación de rostros.
    Arquitectura que coincide exactamente con el modelo guardado original.
    """
    def __init__(self, embedding_size=128):
        super(SiameseNetwork, self).__init__()
        
        # Arquitectura exacta del modelo original (según inspección)
        # - conv1: [64, 3, 3, 3] -> 64 filtros de entrada
        # - conv2: [128, 64, 3, 3] -> 128 filtros
        # - fc1: [512, 401408] -> capa densa grande
        # - fc2: [128, 512] -> embedding final
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Capas completamente conectadas (dimensiones exactas del modelo original)
        self.fc1 = nn.Linear(401408, 512)  # Dimensión exacta del modelo guardado  
        self.fc2 = nn.Linear(512, embedding_size)
        
    def forward_one(self, x):
        """Procesa una imagen a través de la red CNN."""
        # Entrada esperada: [batch_size, 3, 224, 224]
        
        # Capas convolucionales (sin batch normalization como en el modelo original)
        x = self.pool(F.relu(self.conv1(x)))  # [batch, 64, 112, 112]
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 128, 56, 56]
        
        # Aplanar para capas FC
        # 128 * 56 * 56 = 401408 (coincide con fc1 del modelo original)
        x = x.view(x.size(0), -1)
        
        # Capas completamente conectadas
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Embedding final
        
        return x
    
    def forward(self, x):
        """Forward pass de la red."""
        return self.forward_one(x)

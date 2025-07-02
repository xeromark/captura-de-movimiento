import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    """
    Red neuronal siamesa para comparación de rostros.
    Arquitectura simplificada que coincide con el modelo guardado.
    """
    def __init__(self, embedding_size=128):
        super(SiameseNetwork, self).__init__()
        
        # Arquitectura real del modelo guardado (2 capas conv + 2 FC)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Capas completamente conectadas (dimensiones del modelo real)
        self.fc1 = nn.Linear(401408, 512)  # Dimensión exacta del modelo guardado
        self.fc2 = nn.Linear(512, embedding_size)
        
    def forward_one(self, x):
        """Procesa una imagen a través de la red CNN."""
        # Entrada esperada: [batch_size, 3, 224, 224]
        
        # Capas convolucionales (sin batch norm)
        x = self.pool(F.relu(self.conv1(x)))  # 224→112, 3→64
        x = self.pool(F.relu(self.conv2(x)))  # 112→56, 64→128
        
        # Aplanar para capas FC
        x = x.view(x.size(0), -1)
        
        # Capas completamente conectadas
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Embedding final
        
        return x
    
    def forward(self, x):
        """Forward pass de la red."""
        return self.forward_one(x)

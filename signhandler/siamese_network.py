import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    """
    Red neuronal siamesa para comparación de firmas.
    Esta es una definición básica que puede necesitar ajustes según tu modelo específico.
    """
    def __init__(self, input_size=128, hidden_size=64, embedding_size=32):
        super(SiameseNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, embedding_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward_one(self, x):
        """Procesa una entrada a través de la red."""
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def forward(self, x):
        """Forward pass de la red."""
        return self.forward_one(x)

import torch.nn as nn
from typing import List

class TemporalAutoencoder(nn.Module):
    """
    Autoencoder for temporal anomaly detection in time series
    Adapted from SHM damage localization approach
    """

    def __init__(self, input_dim: int, bottleneck_dim: int, hidden_dims: List[int] = None):
        super(TemporalAutoencoder, self).__init__()

        if hidden_dims is None:
            # Default architecture: gradually compress to bottleneck
            hidden_dims = [input_dim // 2, input_dim // 4]

        # Encoder layers
        encoder_layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_dim

        # Bottleneck
        encoder_layers.append(nn.Linear(current_dim, bottleneck_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder layers (mirror of encoder)
        decoder_layers = [nn.Linear(bottleneck_dim, current_dim), nn.ReLU()]

        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_dim

        decoder_layers.append(nn.Linear(current_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

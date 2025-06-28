import torch.nn as nn

class CNNTemporalAutoencoder(nn.Module):
    """
    CNN Autoencoder for temporal anomaly detection in time series
    Uses 1D convolutions to capture temporal patterns
    """

    def __init__(self, sequence_length: int, bottleneck_dim: int = 10, dim_raise: int = 8):
        super(CNNTemporalAutoencoder, self).__init__()

        self.sequence_length = sequence_length

        # Encoder: series of 1D convolutions with pooling
        self.encoder = nn.Sequential(
            # First conv block
            nn.Conv1d(in_channels=1, out_channels=dim_raise, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(dim_raise),
            nn.MaxPool1d(2),

            # Second conv block
            nn.Conv1d(dim_raise, dim_raise*2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(dim_raise*2),
            nn.MaxPool1d(2),

            # Third conv block
            nn.Conv1d(dim_raise*2, dim_raise*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(dim_raise*4),
            nn.MaxPool1d(2),

            # Flatten and bottleneck
            nn.Flatten(),
            nn.Linear(dim_raise*4 * (sequence_length // 8), bottleneck_dim),
            nn.ReLU()
        )

        # Decoder: reconstruct from bottleneck
        self.decoder = nn.Sequential(
            # Expand from bottleneck
            nn.Linear(bottleneck_dim, dim_raise*4 * (sequence_length // 8)),
            nn.ReLU(),

            # Reshape for transposed convolutions
            nn.Unflatten(1, (dim_raise*4, sequence_length // 8)),

            # First transposed conv block
            nn.ConvTranspose1d(dim_raise*4, dim_raise*2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(dim_raise*2),

            # Second transposed conv block
            nn.ConvTranspose1d(dim_raise*2, dim_raise, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(dim_raise),

            # Final transposed conv to original size
            nn.ConvTranspose1d(dim_raise, 1, kernel_size=7, stride=2, padding=3, output_padding=1),
        )

    def forward(self, x):
        # Add channel dimension if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        # Remove channel dimension to match input
        if decoded.shape[1] == 1:
            decoded = decoded.squeeze(1)

        # Ensure output matches input size exactly
        if decoded.shape[-1] != self.sequence_length:
            decoded = nn.functional.interpolate(
                decoded.unsqueeze(1),
                size=self.sequence_length,
                mode='linear',
                align_corners=False
            ).squeeze(1)

        return decoded

    def encode(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        return self.encoder(x)
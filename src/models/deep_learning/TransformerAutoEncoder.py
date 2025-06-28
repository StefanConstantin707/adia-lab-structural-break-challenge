import numpy as np
import torch
import torch.nn as nn
import math

from torch.ao.nn.qat import Conv1d


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels, dtype_override=None):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        :param dtype_override: If set, overrides the dtype of the output embedding.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)
        self.channels = channels
        self.dtype_override = dtype_override

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros(
            (x, self.channels),
            device=tensor.device,
            dtype=(
                self.dtype_override if self.dtype_override is not None else tensor.dtype
            ),
        )
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc


class TransformerConvBlock(nn.Module):
    """Combined transformer and conv block with dimension change"""

    def __init__(self, input_dim: int,
                 output_dim: int,
                 nhead: int = 1,
                 stride: int = 1,
                 conv_kernel: int = 3,
                 padding: int = 0,
                 dropout: float = 0.1
                 ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout

        # Transformer layer
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=input_dim,
            dropout=dropout,
            batch_first=True
        )

        # Dimension change through conv
        self.conv = nn.Conv1d(
            input_dim, output_dim,
            kernel_size=conv_kernel,
            stride=stride,
            padding=padding
        )
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        # x shape: [batch, seq, dim]

        x_transformed = self.transformer_layer(x)  # [batch, seq, dim]

        # Apply conv (need to transpose for conv1d)
        x_conv = x_transformed.transpose(1, 2)  # [batch, dim, seq]
        x_conv = self.conv(x_conv)
        x_conv = self.relu(x_conv)
        x_conv = self.batch_norm(x_conv)

        # Transpose back
        output = x_conv.transpose(1, 2)  # [batch, seq, dim]

        return output


class TransformerDeconvBlock(nn.Module):
    """Combined transformer and transposed conv block with dimension change"""

    def __init__(self, input_dim: int,
                 output_dim: int,
                 nhead: int = 1,
                 stride: int = 1,
                 conv_kernel: int = 3,
                 padding: int = 0,
                 dropout: float = 0.1
                 ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Transformer layer
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=input_dim,
            dropout=dropout,
            batch_first=True
        )

        # Dimension change through transposed conv
        self.conv_transpose = nn.ConvTranspose1d(
            input_dim, output_dim,
            kernel_size=conv_kernel,
            stride=stride,
            padding=padding,
            output_padding=1
        )
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        # x shape: [batch, seq, dim]

        x_transformed = self.transformer_layer(x)  # [batch, dim, seq]

        # Apply transposed conv
        x_conv = x_transformed.transpose(1, 2)  # [batch, dim, seq]
        x_conv = self.conv_transpose(x_conv)
        x_conv = self.relu(x_conv)
        x_conv = self.batch_norm(x_conv)

        # Transpose back
        output = x_conv.transpose(1, 2)  # [batch, seq, dim]

        return output


class TransformerTemporalAutoencoder(nn.Module):
    """
    Progressive Transformer-Conv Autoencoder for temporal anomaly detection
    Interleaves transformer layers with convolutions, gradually increasing dimensions
    """

    def __init__(self, sequence_length: int = 256, bottleneck_dim: int = 16, raise_dim: int = 4, nhead: int = 1, dropout: float = 0.1):
        super(TransformerTemporalAutoencoder, self).__init__()

        self.sequence_length = sequence_length
        self.bottleneck_dim = bottleneck_dim
        self.raise_dim = raise_dim

        self.pos_enc = PositionalEncoding1D(raise_dim)

        # Progressive dimensions: 4 -> 8 -> 16
        dims = [raise_dim, raise_dim * 2, raise_dim * 4]

        self.raise_dim_cov = nn.Conv1d(in_channels=1, out_channels=raise_dim, kernel_size=7, padding=3)

        # Encoder: Progressive transformer-conv blocks with increasing dimensions [64 -> 32]
        self.encoder_block1 = TransformerConvBlock(
            input_dim=dims[0], output_dim=dims[1],
            nhead=nhead, stride=2, conv_kernel=7, dropout=dropout, padding=3
        )

        # [32 -> 16]
        self.encoder_block2 = TransformerConvBlock(
            input_dim=dims[1], output_dim=dims[2],
            nhead=nhead, stride=2, conv_kernel=5, dropout=dropout, padding=2
        )

        # Bottleneck
        compressed_length = self.sequence_length // 4
        self.bottleneck_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dims[2] * compressed_length, bottleneck_dim),
            nn.ReLU()
        )

        # Expand from bottleneck
        self.bottleneck_decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, dims[2] * compressed_length),
            nn.ReLU(),
            nn.Unflatten(1, (compressed_length, dims[2]))
        )

        # Decoder: Progressive transformer-deconv blocks with decreasing dimensions
        self.decoder_block1 = TransformerDeconvBlock(
            input_dim=dims[2], output_dim=dims[1],
            nhead=nhead, stride=2, conv_kernel=5, dropout=dropout, padding=2
        )

        self.decoder_block2 = TransformerDeconvBlock(
            input_dim=dims[1], output_dim=dims[0],
            nhead=nhead, stride=2, conv_kernel=7, dropout=dropout, padding=3
        )

        self.flat = nn.Flatten()

        # Output projection
        self.output_projection = nn.Linear(raise_dim, 1)

    def forward(self, x):

        if x.ndim == 2:
            if x.shape[-1] != 1:
                X = x.unsqueeze(-1)
            else:
                X = x.unsqueeze(0)
        else:
            X = x

        # N, S, C
        batch_size, seq_len, input_dim = X.shape

        if input_dim != 1:
            raise ValueError("Input dimension must be 1")

        x_t = X.transpose(1, 2) # N, S, C -> # N, C, S
        x_raised = self.raise_dim_cov(x_t) # N, 1, S -> # N, raise_dim, S
        x_raised = x_raised.transpose(1, 2) # N, C, S -> N, S, C

        x_raised += self.pos_enc(x_raised) # N, S, raise_dim

        # Encoder path with progressive dimensions
        x1 = self.encoder_block1(x_raised)  # [batch, seq//2, base_dim*2]
        x2 = self.encoder_block2(x1)  # [batch, seq//4, base_dim*4]

        # Bottleneck
        bottleneck = self.bottleneck_encoder(x2)

        # Expand from bottleneck
        expanded = self.bottleneck_decoder(bottleneck)

        # Decoder path with progressive dimensions
        d1 = self.decoder_block1(expanded)
        d2 = self.decoder_block2(d1)  # N, S, raise_dim

        # d2 = self.flat(d2)

        # Output projection: # N, S, raise_dim -> # N, S
        output = self.output_projection(d2).squeeze()

        return output

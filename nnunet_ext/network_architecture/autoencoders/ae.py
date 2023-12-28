#https://github.com/MedVisBonn/Segmentation-Distortion/blob/main/src/model/ae.py
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module
from typing import Iterable, Dict, Callable, Tuple, Union

from nnunet_ext.network_architecture.autoencoders.layers import ConvBlock, ReshapeLayer

    
class AE(nn.Module):
    """Autoencoder (AE) class to transform U-Net feature maps.

    Module that dynamically builds an AE. It expects a certain
    input shape (due to the fc layer). It uses conv blocks
    (see layers.py) and supports different depths, block sizes
    and latent dimensions. Currently, it only supports a dense
    bottleneck.
    """

    def __init__(
        self,
        in_channels: int,
        in_dim: int,
        latent_dim: int = 128,
        depth: int = 3,
        latent: str = "dense",
        block_size: int = 1,
    ):
        super().__init__()

        self.on = True
        self.latent_dim = latent_dim
        self.block_size = block_size

        self.encoder = self._built_encoder(in_channels, in_dim, depth)
        self.intermediate_conv = ConvBlock(
            in_channels * 2**depth,
            in_channels,
            in_dim / (2**depth),
            kernel_size=1,
            stride=1,
            padding=0,
        )

        if latent == "dense":
            self.latent = self._built_latent(in_channels, in_dim, latent_dim, depth)
            self.spatial = self._built_spatial(in_channels, in_dim, latent_dim, depth)

        elif latent == "spatial":
            raise NotImplementedError

        else:
            raise NotImplementedError

        self.intermediate_conv_reverse = ConvBlock(
            in_channels,
            in_channels * 2**depth,
            in_dim / (2**depth),
            kernel_size=1,
            stride=1,
            padding=0,
            reverse=True,
        )
        self.decoder = self._built_decoder(in_channels, in_dim, depth)
        self.tmp = nn.Conv2d(in_channels, in_channels, 1)

    def _built_encoder(self, in_channels: int, in_dim: int, depth: int) -> nn.Module:
        encoder = nn.Sequential(
            *[
                ConvBlock(
                    in_channels * 2**i,
                    in_channels * 2 ** (i + 1),
                    in_dim / (2 ** (i + 1)),
                    block_size=self.block_size,
                    residual=False,
                )
                for i in range(depth)
            ],
        )

        return encoder

    def _built_latent(
        self, in_channels: int, in_dim: int, latent_dim: int, depth: int
    ) -> nn.Module:
        dense_in = int((in_dim / (2**depth)) ** 2 * in_channels)
        dense_out = int(latent_dim)
        latent = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1), nn.Linear(dense_in, dense_out)
        )

        return latent

    def _built_spatial(
        self, in_channels: int, in_dim: int, latent_dim: int, depth: int
    ) -> nn.Module:
        dense_in = int(latent_dim)
        dense_out = int((in_dim / (2**depth)) ** 2 * in_channels)
        channel_in = int(in_channels)
        dim_in = int(in_dim / (2**depth))
        spatial = nn.Sequential(
            nn.Linear(dense_in, dense_out), ReshapeLayer(channel_in, dim_in)
        )

        return spatial

    def _built_decoder(self, in_channels: int, in_dim: int, depth: int) -> nn.Module:
        decoder = nn.Sequential(
            *[
                ConvBlock(
                    in_channels * 2 ** (i + 1),
                    in_channels * 2**i,
                    in_dim / (2**i),
                    reverse=True,
                    padding=1,
                    block_size=self.block_size,
                )
                for i in reversed(range(depth))
            ]
        )

        return decoder

    def turn_off(self):
        self.on = False

    def turn_on(self):
        self.on = True

    def get_latent(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.intermediate_conv(x)
        z = self.latent(x)
        return z

    def forward(self, x: Tensor) -> Tensor:
        """Either applies the AE to the input or
        simply the identity function. This is a useful
        utility for debugging and experimentation. It can
        be turned off and on via the turn_off and turn_on
        class methods
        """

        if self.on:
            x = self.encoder(x)
            z = self.intermediate_conv(x)
            z = self.latent(z)
            z = self.spatial(z)
            z = self.intermediate_conv_reverse(z)
            z = self.decoder(x + z)
            z = self.tmp(z)
            return z

        else:
            return x

        
class ChannelAE(nn.Module):
    """Autoencoder (AE) class to transform U-Net feature maps.

    Module that dynamically builds an AE. It expects a certain
    input shape (due to the fc layer). It uses conv blocks
    (see layers.py) and supports different depths, block sizes
    and latent dimensions. Currently, it only supports a dense
    bottleneck.
    """
    
    def __init__(
        self,
        in_channels: int,
        in_dim: int,
        depth: int = 3,
        block_size: int = 1,
        residual: bool = False
    ):
        super().__init__()
        self.on = True
        self.in_channels = in_channels
        self.in_dim = in_dim
        self.depth = depth
        self.block_size = block_size
        self.residual = residual
        
        
        self.init = ConvBlock(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            in_dim=self.in_dim,
            block_size=self.block_size,
            residual=self.residual,
            kernel_size=3,
            stride=1,
            padding=1,)
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.out     = nn.Conv2d(in_channels, in_channels, 1)
        
        
        
#     def _build_encoder(self):
#         encoder_list = nn.ModuleList(
#             ConvBlock(
#                 in_channels=self.in_channels // 4**i,
#                 out_channels=self.in_channels // 4**(i+1),
#                 in_dim=self.in_dim,
#                 block_size=self.block_size,
#                 residual=self.residual,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#             ) for i in range(self.depth)
#         )
#         return encoder_list 

    
#     def _build_decoder(self):
#         decoder_list = nn.ModuleList(
#             ConvBlock(
#                 in_channels=self.in_channels // 4**(i+1),
#                 out_channels=self.in_channels // 4**i,
#                 in_dim=self.in_dim,
#                 block_size=self.block_size,
#                 residual=self.residual,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#             ) for i in reversed(range(self.depth))
#         )
#         return decoder_list
    
    
    def _build_encoder(self):
        encoder_list = nn.ModuleList(
            ConvBlock(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                in_dim=self.in_dim,
                block_size=self.block_size,
                residual=self.residual,
                kernel_size=3,
                stride=1,
                padding=1,
            ) for i in range(self.depth)
        )
        return encoder_list 

    
    def _build_decoder(self):
        decoder_list = nn.ModuleList(
            ConvBlock(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                in_dim=self.in_dim,
                block_size=self.block_size,
                residual=self.residual,
                kernel_size=3,
                stride=1,
                padding=1,
            ) for i in reversed(range(self.depth))
        )
        return decoder_list
    
    
    def forward(self, x):
        encoder_outputs = []
        
        # Encoding
        x = self.init(x)
        encoder_outputs.append(x)
        for enc_layer in self.encoder:
            x = enc_layer(x)
            encoder_outputs.append(x)

        # Decoding
        for i, dec_layer in enumerate(self.decoder):
            if i > 0:
                x = dec_layer(x) + encoder_outputs[-(i + 2)]  # Skip connection
            else:
                x = dec_layer(x) # no skip connection if bottleneck
        # Output layer
        x = self.out(x)
        return x
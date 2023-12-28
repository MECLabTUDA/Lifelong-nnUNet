#https://github.com/MedVisBonn/Segmentation-Distortion/blob/main/src/model/layers.py
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module
from typing import Iterable, Dict, Callable, Tuple, Union
from itertools import chain


class ConvBlock(nn.Module):
    """Conv block for the AE model.

    Dynamic conv block that supports both down and up
    convolutions as well as different block sizes. It's
    the main building block for the auto encoder.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_dim: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        block_size: int = 1,
        reverse: bool = False,
        residual: bool = False
    ):
        super().__init__()
        self.block_size = block_size
        self.residual = residual

        in_channels = int(in_channels)
        out_channels = int(out_channels)
        in_dim = int(in_dim)

        if not reverse:
            if block_size > 1:
                self.block = nn.Sequential(
                    *chain.from_iterable(
                        [
                            [
                                nn.Conv2d(
                                    in_channels,
                                    in_channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=padding,
                                ),
                                nn.LayerNorm(
                                    # little hacky, works for stride 1 and 2
                                    torch.Size([in_channels, in_dim * stride, in_dim * stride])
                                ),
                                nn.LeakyReLU(),
                            ]
                            for _ in range(block_size - 1)
                        ]
                    )
                )

            self.sample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
                nn.LayerNorm(torch.Size([out_channels, in_dim, in_dim])),
                nn.LeakyReLU(),
            )

        else:
            if padding == 0:
                output_padding = 0
            else:
                output_padding = 1

            if block_size > 1:
                self.block = nn.Sequential(
                    *chain.from_iterable(
                        [
                            [
                                nn.ConvTranspose2d(
                                    in_channels,
                                    in_channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=padding,
                                    output_padding=0,
                                ),
                                nn.LayerNorm(
                                    torch.Size([in_channels, in_dim // 2, in_dim // 2])
                                ),
                                nn.LeakyReLU(),
                            ]
                            for _ in range(block_size - 1)
                        ]
                    )
                )

            self.sample = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                ),
                nn.LayerNorm(torch.Size([out_channels, in_dim, in_dim])),
                nn.LeakyReLU(),
            )

    def forward(self, x: Tensor) -> Tensor:
        if self.block_size > 1:
            x = x + self.block(x) if self.residual else self.block(x)
        return self.sample(x)





class ReshapeLayer(nn.Module):
    """Reshape layer to transform vectors into spatial feature maps.

    This layer is used to transform flattened representations into spatial
    feature maps that can be used by regular convolution layers. It's used
    as the first layer in the AE decoder, after the fc layer.
    """

    def __init__(self, in_channels, in_dim):
        super().__init__()

        self.in_channels = int(in_channels)
        self.in_dim = int(in_dim)

    def forward(self, x: Tensor) -> Tensor:
        return x.view(-1, self.in_channels, self.in_dim, self.in_dim)

    

class DuplicateIdentity(nn.Module):
    """Identity layer that has the same output signature as a VAE.
    
    This is used to attach hooks that expect a certain output structure.
    Its also useful when the layer we want to attach to splits its output
    into multiple paths, e.g. right before a skip connection in a UNet.
    """
    def __init__(self, n_samples=10):
        super().__init__()
        
    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        return None, None, x_in
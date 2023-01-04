
import numpy as np
from batchgenerators.augmentations.utils import pad_nd_image
from nnunet.utilities.random_stuff import no_op
from nnunet.utilities.to_torch import to_cuda, maybe_to_torch
from torch import nn
import torch
from scipy.ndimage.filters import gaussian_filter
from typing import Union, Tuple, List, Sequence, Optional, Any
from torch.cuda.amp import autocast
from torch import nn, Tensor
import torch

from nnunet_ext.network_architecture.superclasses.autoencoder import Autoencoder


class expert_gate_autoencoder(Autoencoder):

    
    def __init__(self, experiment: str) -> None:
        super().__init__()
        
        if experiment in ["expert_gate_simple_ae"]:
            self.encoder = torch.nn.Sequential(
                nn.Conv2d(1,3,5,padding="same"),
                nn.Sigmoid()
            )
            self.decoder = torch.nn.Sequential(
                nn.Conv2d(3,1,5,padding="same")
            )
        elif experiment in ["expert_gate_simple_ae_alex_features"]:
            self.encoder = torch.nn.Sequential(
                nn.Conv2d(256,256,3,padding="same"),
                nn.Sigmoid()
            )
            self.decoder = torch.nn.Sequential(
                nn.Conv2d(256,256,3,padding="same")
            )
        elif experiment in ["expert_gate_simple_ae_UNet_features"]:
            self.encoder = torch.nn.Sequential(
                nn.Conv3d(32,3,5,padding="same"),
                nn.Sigmoid()
            )
            self.decoder = torch.nn.Sequential(
                nn.Conv3d(3,32,5,padding="same")
            )
        else:
            raise NotImplementedError("did not expect expert_gate_autoencoder to be constructed with experiment: " + experiment)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x


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

import traceback

class expert_gate_autoencoder(Autoencoder):

    
    def __init__(self) -> None:
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            nn.Conv2d(1,16,3,padding="same"),
            nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            #nn.ConvTranspose2d(16,1,3,padding=3)
            nn.Conv2d(16,1,3,padding="same")
        )

        if torch.cuda.is_available():
            self.to('cuda')

    def forward(self, x: Tensor) -> Tensor: #TODO make sure that the input are features extracted by alexnet
        x = self.encoder(x)
        x = self.decoder(x)
        return x

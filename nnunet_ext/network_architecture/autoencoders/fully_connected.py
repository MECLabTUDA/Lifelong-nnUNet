import numpy as np
import torch
import torch.nn as nn

class FullyConnectedAE(nn.Module):
    def __init__(self, shape) -> None:
        super().__init__()
        assert len(shape) == 3, shape
        num_dimensions = np.prod(shape)
        self.encoder = nn.Sequential(
                                nn.Linear(num_dimensions, int(0.75 * num_dimensions)),
                                nn.BatchNorm1d(int(0.75 * num_dimensions)),
                                nn.LeakyReLU(),
                                nn.Linear(int(0.75 * num_dimensions), int(0.5 * num_dimensions)),
                                nn.BatchNorm1d(int(0.5 * num_dimensions)),
                                nn.LeakyReLU())
        
        self.decoder = nn.Sequential(nn.Linear(int(0.5 * num_dimensions), int(0.75 * num_dimensions)),
                                nn.BatchNorm1d(int(0.75 * num_dimensions)),
                                nn.LeakyReLU(),
                                #nn.Linear(int(0.8 * num_dimensions), int(0.8 * num_dimensions)),
                                #nn.BatchNorm1d(int(0.8 * num_dimensions)),
                                #nn.LeakyReLU(),
                                nn.Linear(int(0.75 * num_dimensions), num_dimensions),
                                nn.Unflatten(1, shape),
                                nn.LeakyReLU()
                                )
        
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def to_gpus(self):
        self.cuda(0)
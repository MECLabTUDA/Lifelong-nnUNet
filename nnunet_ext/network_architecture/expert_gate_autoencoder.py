
from torch import nn, Tensor
import torch

class expert_gate_autoencoder(nn.Module):
    autoencoder: nn.Sequential

    
    def __init__(self, input_dims: int, code_dims: int = None) -> None:
        super().__init__()
        if code_dims == None:
            #code_dims = int(0.003 * input_dims)
            #code_dims = int(0.04 * input_dims)
            code_dims = int(0.1 * input_dims)
        
        self.autoencoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dims,code_dims),
            nn.ReLU(),
            nn.Linear(code_dims, input_dims),
            nn.Sigmoid()
        )
        if torch.cuda.is_available():
            self.to('cuda')

    def forward(self, x: Tensor) -> Tensor: #TODO make sure that the input are features extracted by alexnet
        shape = x.shape
        x  = self.autoencoder(x)
        x = torch.reshape(x, shape)
        return x

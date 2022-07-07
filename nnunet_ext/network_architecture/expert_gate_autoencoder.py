
from torch import nn, Tensor


class expert_gate_autoencoder(nn.Module):
    autoencoder: nn.Sequential

    
    def __init__(self, input_dims: int, code_dims: int = 100) -> None:
        super().__init__()
        self.autoencoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dims,code_dims),
            nn.ReLU(),
            nn.Linear(code_dims, input_dims),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor: #TODO make sure that the input are features extracted by alexnet
        return self.autoencoder(x)
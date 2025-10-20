from matplotlib import pyplot as plt
import torch, einops
import torch.nn as nn
import torch.nn.functional as F
from nnunet_ext.network_architecture.nca.NCA2D import NCA2D

def merge_ncas(base_nca: NCA2D, new_nca: NCA2D) -> NCA2D:
    device = base_nca.conv.weight.device
    hidden_size0 = base_nca.hidden_size
    hidden_size1 = new_nca.hidden_size

    merged_nca = NCA2D(num_channels=base_nca.num_channels + new_nca.num_channels, num_input_channels=base_nca.num_input_channels, 
                       num_classes=16, hidden_size=hidden_size0 + hidden_size1, fire_rate=1, num_steps=10, use_norm=False).to(device)

    merged_nca.conv.weight = torch.nn.Parameter(torch.cat([base_nca.conv.weight, new_nca.conv.weight], dim=0))
    merged_nca.conv.bias = torch.nn.Parameter(torch.cat([base_nca.conv.bias, new_nca.conv.bias], dim=0))

    base_nca_fc0_upper = base_nca.fc0.weight[:, :base_nca.num_channels, :, :]
    base_nca_fc0_lower = base_nca.fc0.weight[:, base_nca.num_channels:, :, :]

    base_nca_fc0_w_zeros = torch.cat([base_nca_fc0_upper, torch.zeros(hidden_size0, new_nca.num_channels, 1, 1, device=device),
                                      base_nca_fc0_lower, torch.zeros(hidden_size0, new_nca.num_channels, 1, 1, device=device)], dim=1)
    fc0 = torch.cat([base_nca_fc0_w_zeros, new_nca.fc0.weight], dim=0)
    assert merged_nca.fc0.weight.shape == fc0.shape, f"{merged_nca.fc0.weight.shape}, {base_nca_fc0_w_zeros.shape}"
    merged_nca.fc0.weight = torch.nn.Parameter(fc0)
    merged_nca.fc0.bias = torch.nn.Parameter(torch.cat([base_nca.fc0.bias, new_nca.fc0.bias], dim=0))


    base_nca_fc1_w_zeros = torch.cat([base_nca.fc1.weight, torch.zeros(base_nca.num_channels-base_nca.num_input_channels, hidden_size1, 1, 1, device=device)], dim=1)
    new_nca_fc1_w_zeros = torch.cat([torch.zeros(new_nca.num_channels, hidden_size0, 1, 1, device=device), new_nca.fc1.weight], dim=1)
    fc1 = torch.cat([base_nca_fc1_w_zeros, new_nca_fc1_w_zeros], dim=0)
    assert merged_nca.fc1.weight.shape == fc1.shape, f"{merged_nca.fc1.weight.shape}, {fc1.shape}"
    merged_nca.fc1.weight = torch.nn.Parameter(fc1)
    assert base_nca.fc1.bias is None and new_nca.fc1.bias is None
    assert merged_nca.fc1.bias is None
    return merged_nca


class LargeNCA2D(nn.Module):
    def __init__(self, base_nca: NCA2D, num_channels: int, num_new_classes: int,
                 hidden_size: int):
        super(LargeNCA2D, self).__init__()
        self.base_nca = base_nca
        self._init_new_nca(num_channels, num_new_classes, hidden_size)


        self.base_nca.requires_grad_(False)

    def _nca_update_extended(self, states: list[torch.Tensor]):
        delta_state0 = self.base_nca.conv(states[0])
        delta_state1 = self.new_nca.conv(states[1])
        delta_state1 = torch.cat([states[0], states[1], delta_state0, delta_state1], dim=1)
        delta_state0 = torch.cat([states[0], delta_state0], dim=1)
        delta_state0 = self.base_nca.fc0(delta_state0)
        delta_state1 = self.new_nca.fc0(delta_state1)
        delta_state0 = self.base_nca.batch_norm(delta_state0)
        delta_state1 = self.new_nca.batch_norm(delta_state1)
        delta_state0 = F.relu(delta_state0, inplace=True)
        delta_state1 = F.relu(delta_state1, inplace=True)
        delta_state0 = self.base_nca.fc1(delta_state0)
        delta_state1 = self.new_nca.fc1(delta_state1)

        if self.base_nca.fire_rate < 1.0:
            with torch.no_grad():
                stochastic = torch.zeros((delta_state0.shape[0], 1, delta_state0.shape[2], delta_state0.shape[3]), device=delta_state0.device)
                stochastic.bernoulli_(self.base_nca.fire_rate).float()
            delta_state0 = delta_state0 * stochastic
            delta_state1 = delta_state1 * stochastic

        temp_state0 = states[0][:, self.base_nca.num_input_channels:] + delta_state0
        temp_state0 = torch.cat([states[0][:, :self.base_nca.num_input_channels], temp_state0], dim=1)
        return [temp_state0, states[1] + delta_state1]
        

    def make_state(self, x: torch.Tensor) -> list[torch.Tensor]:
        state1 = torch.zeros(x.shape[0], self.base_nca.num_channels - self.base_nca.num_input_channels,
                    x.shape[2], x.shape[3], device=x.device)
        state2 = torch.zeros(x.shape[0], self.new_nca.num_channels, x.shape[2], x.shape[3], device=x.device)
        states = [torch.cat([x, state1], dim=1), state2]
        return states

    def forward_internal(self, states: list[torch.Tensor]) -> list[torch.Tensor]:
        for _ in range(self.base_nca.num_steps):
            states = self._nca_update_extended(states)
        return states

    def forward(self, x):
        raise NotImplementedError("LargeNCA2D does not implement forward directly. Use _forward_extended instead.")
    
    def _init_new_nca(self, num_channels: int, num_new_classes: int,
                 hidden_size: int):
        assert not hasattr(self, 'new_nca'), "New NCA already exists."
        self.new_nca = NCA2D(num_channels=num_channels,
                             num_input_channels=0,
                             num_classes=num_new_classes,
                             hidden_size=hidden_size,
                             fire_rate=self.base_nca.fire_rate,
                             num_steps=self.base_nca.num_steps,
                             use_norm=not isinstance(self.base_nca.batch_norm, nn.Identity))
        self.new_nca.fc0 = nn.Conv2d(2*(self.base_nca.num_channels + self.new_nca.num_channels),
                                        hidden_size, kernel_size=1)
        self.new_nca.to(self.base_nca.conv.weight.device)
        
    def merge_and_init_new_nca(self, num_channels: int, num_new_classes: int,
                 hidden_size: int):
        self.base_nca = merge_ncas(self.base_nca, self.new_nca)
        self.base_nca.requires_grad_(False)

        del self.new_nca
        self._init_new_nca(num_channels, num_new_classes, hidden_size)
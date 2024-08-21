from matplotlib import pyplot as plt
import torch, einops
import torch.nn as nn
import torch.nn.functional as F

class NCA2D(nn.Module):
    def __init__(self, num_channels: int, num_input_channels: int, num_classes: int,
                 hidden_size: int, fire_rate: float, num_steps: int):
        super(NCA2D, self).__init__()
        self.do_ds = False


        self.fc0 = nn.Linear(2 * num_channels, hidden_size)
        self.fc1 = nn.Linear(hidden_size, num_channels - num_input_channels, bias=False)

        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding='same', 
                              padding_mode="reflect", groups=num_channels)
        self.batch_norm = nn.BatchNorm2d(hidden_size, track_running_stats=False)

        #with torch.no_grad():
        #    self.fc0.weight.zero_()

        self.fire_rate = fire_rate
        self.num_channels = num_channels
        self.num_input_channels = num_input_channels
        self.num_steps = num_steps
        self.num_classes = num_classes

    def update(self, state):
        delta_state = self.conv(state)
        delta_state = torch.cat([state, delta_state], dim=1)
        delta_state = einops.rearrange(delta_state, 'b c h w -> b h w c')
        delta_state = self.fc0(delta_state)
        delta_state = einops.rearrange(delta_state, 'b h w c -> b c h w')
        delta_state = self.batch_norm(delta_state)
        delta_state = einops.rearrange(delta_state, 'b c h w -> b h w c')
        delta_state = F.relu(delta_state, inplace=True)
        delta_state = self.fc1(delta_state)

        stochastic = torch.rand((delta_state.shape[0], delta_state.shape[1], delta_state.shape[2], 1), 
                                device=delta_state.device)
        stochastic = stochastic > 1 - self.fire_rate
        stochastic = stochastic.float()
        delta_state = delta_state * stochastic

        delta_state = einops.rearrange(delta_state, 'b h w c -> b c h w')

        return state[:, self.num_input_channels:] + delta_state
        

    def make_state(self, x):
        state = torch.zeros(x.shape[0], self.num_channels - self.num_input_channels,
                            x.shape[2], x.shape[3], device=x.device)
        state = torch.cat([x, state], dim=1)
        return state

    def forward_internal(self, state, show=False):
        for _ in range(self.num_steps):
            new_state = self.update(state)
            if show:
                plt.subplot(1,self.num_steps,_+1)
                plt.imshow(new_state.detach().cpu().numpy()[0,0])
            state = torch.cat([state[:, :self.num_input_channels], new_state], dim=1)
        if show:
            plt.show()
        return state

    def forward(self, x):
        state = self.make_state(x)
        return self.forward_internal(state)[:, self.num_input_channels:self.num_input_channels + self.num_classes]
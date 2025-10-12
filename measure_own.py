from nnunet_ext.network_architecture.nca.NCA2D import NCA2D
import torch, pickle
import torch.nn.functional as F

torch.cuda.memory._record_memory_history()

device = "cuda"

nca1 = NCA2D(num_channels=16, num_input_channels=1, num_classes=16, hidden_size=32, fire_rate=1, num_steps=10)
nca1.to(device)

x = torch.randn(1,1,128,128, device=device)

hidden_size2 = 4* 3
nca2 = NCA2D(num_channels=3, num_input_channels=0, num_classes=16, hidden_size=hidden_size2, fire_rate=1, num_steps=10)
nca2.fc0 = torch.nn.Conv2d(2 * (nca1.num_channels + nca2.num_channels), hidden_size2, kernel_size=1)
nca2.to(device)

nca1.requires_grad_(False)

#before = torch.cuda.memory.memory_allocated()
state1 = torch.zeros(x.shape[0], nca1.num_channels - nca1.num_input_channels,
                    x.shape[2], x.shape[3], device=x.device)
state2 = torch.zeros(x.shape[0], nca2.num_channels, x.shape[2], x.shape[3], device=x.device)
state = [torch.cat([x, state1], dim=1), state2]
del state1

def step(state: list[torch.Tensor], nca1: NCA2D, nca2: NCA2D) -> list[torch.Tensor]:
    delta_state0 = nca1.conv(state[0])
    delta_state1 = nca2.conv(state[1])
    delta_state0 = torch.cat([state[0], delta_state0], dim=1)
    delta_state1 = torch.cat([delta_state0, state[1], delta_state1], dim=1)
    delta_state0 = nca1.fc0(delta_state0)
    delta_state1 = nca2.fc0(delta_state1)
    delta_state0 = nca1.batch_norm(delta_state0)
    delta_state1 = nca2.batch_norm(delta_state1)
    delta_state0 = F.relu(delta_state0, inplace=True)
    delta_state1 = F.relu(delta_state1, inplace=True)
    delta_state0 = nca1.fc1(delta_state0)
    delta_state1 = nca2.fc1(delta_state1)
    temp_state0 = state[0][:, nca1.num_input_channels:] + delta_state0
    temp_state0 = torch.cat([state[0][:, :nca1.num_input_channels], temp_state0], dim=1)
    state[0] = temp_state0
    state[1] = state[1] + delta_state1
    return state

for _ in range(2):
    state = step(state, nca1, nca2)
#(state[0].sum() + state[1].sum()).backward()
#after = torch.cuda.memory.memory_allocated()
#print(f"{(after-before) / 1024**2:.2f} MB allocated during baseline training step")


torch.cuda.memory._dump_snapshot("snapshot_own.pickle")
#with open("snapshot_own.pickle", "wb") as f:
#    pickle.dump(torch.cuda.memory._snapshot(), f)
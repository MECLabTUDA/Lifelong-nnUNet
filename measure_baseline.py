from nnunet_ext.network_architecture.nca.NCA2D import NCA2D
import torch
import torch.nn.functional as F

device = "cuda"

nca1 = NCA2D(num_channels=16, num_input_channels=1, num_classes=16, hidden_size=32, fire_rate=1, num_steps=10)
nca1.to(device)

x = torch.randn(1,1,128,128, device=device)

before = torch.cuda.memory.memory_allocated()
state = nca1.make_state(x)
out = nca1.forward_internal(state)
out.sum().backward()
after = torch.cuda.memory.memory_allocated()
print(f"{(after-before) / 1024**2:.2f} MB allocated during baseline training step")
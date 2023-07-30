from nnunet_ext.network_architecture.VAE import VAE
from matplotlib import pyplot as plt
import torch
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda


def visualize_latent_space(vae: VAE, dataloader, out_path: str, num_batches: int=10):
    data_iter = iter(dataloader)
    for _ in range(num_batches):
        try:
            data_dict = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            data_dict = next(data_iter)

        x = data_dict['data'][-1]
        x = maybe_to_torch(x)
        if torch.cuda.is_available():
            x = to_cuda(x)

        mean, log_var = vae.encode(x)
        z = vae.sample_from(mean, log_var).detach().cpu().numpy()

        plt.scatter(z[:,0],z[:,1], color="red")
        plt.savefig(out_path)



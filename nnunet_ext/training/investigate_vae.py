from nnunet_ext.network_architecture.VAE import VAE, SecondStageVAE
from matplotlib import pyplot as plt
import torch
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda

from nnunet_ext.training.FeatureRehearsalDataset import InfiniteIterator

@torch.no_grad()
def visualize_latent_space(vae: VAE, dataloader, out_path: str, num_batches: int=10):
    vae = vae.cuda().eval()
    plt.clf()
    data_iter = InfiniteIterator(dataloader)
    for _ in range(num_batches):
        data_dict = next(data_iter)

        x = data_dict['data'][-1]
        x = maybe_to_torch(x)
        if torch.cuda.is_available():
            x = to_cuda(x)

        mean, log_var = vae.encode(x)
        z = vae.sample_from(mean, log_var).detach().cpu().numpy()

        plt.scatter(z[:,0],z[:,1], color="red")
        plt.savefig(out_path)

@torch.no_grad()
def visualize_second_stage_latent_space(first_stage_vae: VAE, second_stage_vae: SecondStageVAE, dataloader, out_path: str, num_batches: int=10):
    colorMap = ["red","green","blue","yellow","pink","magenta","orange","purple","beige","brown","gray","cyan","black"]
    plt.clf()
    data_iter = InfiniteIterator(dataloader)
    for _ in range(num_batches):
        data_dict = next(data_iter)

        x = data_dict['data'][-1]
        x = maybe_to_torch(x)
        task_idx = data_dict['task_idx']
        if torch.cuda.is_available():
            x = to_cuda(x)
            task_idx = to_cuda(task_idx)

        mean_u, log_var_u =  first_stage_vae.encode(x)
        u = first_stage_vae.sample_from(mean_u, log_var_u)
        mean_z, log_var_z = second_stage_vae.encode(u,task_idx)
        z = second_stage_vae.sample_from(mean_z, log_var_z).detach().cpu().numpy()

        task_idx = task_idx.cpu().numpy()

        for t in range(max(task_idx)+1):
            temp_z = z[task_idx == t]
            plt.scatter(temp_z[:,0], temp_z[:,1], color=colorMap[t])
        
        plt.savefig(out_path)



import torch
import torch.nn as nn
import numpy as np

# inspired by https://github.com/chendaichao/VAE-pytorch


class Unflatten(nn.Module):
    def forward(self, x):
        assert len(x.shape)==2  #x: (B, C)
        return x[:, :, None, None]
    
class Resize(nn.Module):
    def __init__(self, shape) -> None:
        super().__init__()
        self.shape = shape
        assert len(self.shape) == 3
    def forward(self, x):
        assert len(x.shape) == len(self.shape)+1, f"{len(x.shape)} == {len(self.shape)}+1"
        assert x.shape[1] == self.shape[0]#<- usually, we do not want to change the num of channels

        left_x = int(np.ceil((x.shape[2] - self.shape[1])/2))
        right_x = x.shape[2] - int(np.floor((x.shape[2] - self.shape[1])/2))

        left_y = int(np.ceil((x.shape[3] - self.shape[2])/2))
        right_y = x.shape[3] - int(np.floor((x.shape[3] - self.shape[2])/2))

        return x[:, :, left_x:right_x, left_y:right_y]


class VAE(nn.Module):
    def __init__(self, shape, hidden_dim) -> None:
        super().__init__()

        #for now:
        assert len(shape) == 3  
        # shape: (C, H, W)

        num_channels =  shape[0]
        assert hidden_dim >= shape[0]

        shape = np.asarray(list(shape))

        encoder = []
        decoder = [Resize(shape),nn.Conv2d(hidden_dim, num_channels,3,1,1)]
        while np.any(shape[1:] >1):
            stride = np.asarray([1,1])
            stride[shape[1:] >1] = 2

            encoder.append(nn.Conv2d(num_channels,hidden_dim,(3,3), stride ,1))
            encoder.append(nn.BatchNorm2d(hidden_dim))
            encoder.append(nn.LeakyReLU(inplace=True))
            encoder.append(nn.Conv2d(hidden_dim,hidden_dim,(3,3),1,1))
            encoder.append(nn.BatchNorm2d(hidden_dim))
            encoder.append(nn.LeakyReLU(inplace=True))


            decoder.append(nn.LeakyReLU(inplace=True))
            decoder.append(nn.BatchNorm2d(hidden_dim))
            decoder.append(nn.Conv2d(hidden_dim,hidden_dim,(3,3),1,1))
            decoder.append(nn.LeakyReLU(inplace=True))
            decoder.append(nn.BatchNorm2d(hidden_dim))
            decoder.append(nn.Conv2d(hidden_dim,hidden_dim,(3,3),1,1))
            decoder.append(nn.ConvTranspose2d(hidden_dim, hidden_dim, stride, stride, bias=False))


            shape = np.ceil(shape /2)
            num_channels=hidden_dim

        #self.encoder = nn.Conv2d(shape[0], hidden_dim,3,2,1)
        encoder.append(nn.Flatten())
        encoder.append(nn.Linear(hidden_dim,hidden_dim))
        encoder.append(nn.BatchNorm1d(hidden_dim))
        encoder.append(nn.LeakyReLU(inplace=True))
        self.encoder = nn.Sequential(*encoder)

        decoder.append(Unflatten())
        decoder.append(nn.LeakyReLU(inplace=True))
        decoder.append(nn.BatchNorm1d(hidden_dim))
        decoder.append(nn.Linear(hidden_dim,hidden_dim))
        self.decoder = nn.Sequential(*(decoder[::-1]))

        self.compute_mean = nn.Linear(hidden_dim, hidden_dim)
        self.compute_log_var = nn.Linear(hidden_dim, hidden_dim)

    def sample_from(self, mean, log_var):
        eps = torch.randn(mean.shape, device=mean.device)
        var = 0.5 * torch.exp(log_var)
        return mean + eps * var

    def encode(self, x):
        x = self.encoder(x)
        return self.compute_mean(x), self.compute_log_var(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, log_var =self.encode(x)

        z = self.sample_from(mean, log_var)
        x_hat = self.decode(z)
        return x_hat, mean, log_var
    

class SecondStageVAE(nn.Module):
    def __init__(self, input_dim: int, dim_of_conditional: int, num_tasks:int) -> None:
        super().__init__()

        self.task_embedding = nn.Embedding(num_tasks, dim_of_conditional)

        self.latent_dim = input_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim + dim_of_conditional, 2* input_dim),
            nn.BatchNorm1d(2* input_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2* input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.LeakyReLU(inplace=True)
        )
        self.compute_mean = nn.Linear(input_dim, input_dim)
        self.compute_log_var = nn.Linear(input_dim, input_dim)

        self.decoder = nn.Sequential(
            nn.Linear(input_dim + dim_of_conditional, 2* input_dim),
            nn.BatchNorm1d(2* input_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2* input_dim, input_dim)
        )

    def sample_from(self, mean, log_var):
        eps = torch.randn(mean.shape, device=mean.device)
        var = 0.5 * torch.exp(log_var)
        return mean + eps * var

    def encode(self, x, task_idx: int):
        y = self.task_embedding(task_idx)
        x = torch.cat((x,y), dim=1)
        x = self.encoder(x)
        return self.compute_mean(x), self.compute_log_var(x)
    
    def decode(self, z, task_idx):
        y = self.task_embedding(task_idx)
        z = torch.cat((z,y), dim=1)
        return self.decoder(z)


    def forward(self, x, task_idx: int):
        mean, log_var = self.encode(x, task_idx)

        z = self.sample_from(mean, log_var)
        x_hat = self.decode(z, task_idx)
        return x_hat, mean, log_var
    
    def generate(self, task_idx: int or torch.Tensor):
        if (type(task_idx) is int):
            task_idx = torch.tensor(task_idx)
        task_idx = task_idx.cuda()
        if (len(task_idx.shape) == 0):
            batch_size = None
            class_idx = class_idx.unsqueeze(0)
            z = torch.randn((1, self.latent_dim)).cuda()
        else:
            batch_size = class_idx.shape[0]
            z = torch.randn((batch_size, self.latent_dim)).cuda()

        
        y = self.task_embedding(task_idx)
        res = self.decode(z, y)
        if not batch_size:
            res = res.squeeze(0)
        return res
    
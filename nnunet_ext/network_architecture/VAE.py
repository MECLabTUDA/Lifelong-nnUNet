from collections import OrderedDict
from typing import Any
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

#import torch.distributed.fsdp as fsdp

# inspired by https://github.com/chendaichao/VAE-pytorch

def lossMSE(X, X_hat, mean, logvar):
    reconstruction_loss = torch.sum(F.mse_loss(X_hat, X, reduction="none"), dim=tuple(range(1,len(X_hat.shape))))
    KL_divergence = torch.sum(0.5 * (-1 - logvar + torch.exp(logvar) + mean**2), dim=tuple(range(1, len(mean.shape))))
    
    return reconstruction_loss, KL_divergence

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


class InitWeightsVAE():
    def __init__(self, method: str):
        assert method in ["none", "log", "sqrt"]
        if method == "none":
            self.method = lambda x : x 
        elif method == "log":
            self. method = lambda x : torch.log(x)
        elif method == "sqrt":
            self.method = lambda x : torch.sqrt(x)

    def __call__(self, module):
        if isinstance(module, nn.Linear):
            stdv = 1. / self.method(module.weight.size(1))
            module.weight = nn.init._no_grad_uniform_(-stdv, stdv)
            if module.bias is not None:
                module.bias = nn.init._no_grad_uniform_(-stdv, stdv)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)
class Encoder(nn.Module):
    def __init__(self, shape, nhid = 16, ncond = 0):
        super(Encoder, self).__init__()
        c, h, w = shape
        ww = ((w-8)//2 - 4)//2
        hh = ((h-8)//2 - 4)//2
        self.encode = nn.Sequential(nn.Conv2d(c, 16, 5, padding = 0), nn.BatchNorm2d(16), nn.ReLU(inplace = True), 
                                    nn.Conv2d(16, 32, 5, padding = 0), nn.BatchNorm2d(32), nn.ReLU(inplace = True), 
                                    nn.MaxPool2d(2, 2),
                                    nn.Conv2d(32, 64, 3, padding = 0), nn.BatchNorm2d(64), nn.ReLU(inplace = True), 
                                    nn.Conv2d(64, 64, 3, padding = 0), nn.BatchNorm2d(64), nn.ReLU(inplace = True), 
                                    nn.MaxPool2d(2, 2),
                                    Flatten(), MLP([ww*hh*64, 256, 128])
                                   )
        self.calc_mean = MLP([128+ncond, 64, nhid], last_activation = False)
        self.calc_logvar = MLP([128+ncond, 64, nhid], last_activation = False)
    def forward(self, x, y = None):
        x = self.encode(x)
        if (y is None):
            return self.calc_mean(x), self.calc_logvar(x)
        else:
            return self.calc_mean(torch.cat((x, y), dim=1)), self.calc_logvar(torch.cat((x, y), dim=1))
class Decoder(nn.Module):
    def __init__(self, shape, nhid = 16, ncond = 0):
        super(Decoder, self).__init__()
        c, w, h = shape
        self.shape = shape
        self.decode = nn.Sequential(MLP([nhid+ncond, 64, 128, 256, c*w*h], last_activation = False), nn.Sigmoid())
    def forward(self, z, y = None):
        c, w, h = self.shape
        if (y is None):
            return self.decode(z).view(-1, c, w, h)
        else:
            return self.decode(torch.cat((z, y), dim=1)).view(-1, c, w, h)
class MLP(nn.Module):
    def __init__(self, hidden_size, last_activation = True):
        super(MLP, self).__init__()
        q = []
        for i in range(len(hidden_size)-1):
            in_dim = hidden_size[i]
            out_dim = hidden_size[i+1]
            q.append(("Linear_%d" % i, nn.Linear(in_dim, out_dim)))
            if (i < len(hidden_size)-2) or ((i == len(hidden_size) - 2) and (last_activation)):
                q.append(("BatchNorm_%d" % i, nn.BatchNorm1d(out_dim)))
                q.append(("ReLU_%d" % i, nn.ReLU(inplace=True)))
        self.mlp = nn.Sequential(OrderedDict(q))
    def forward(self, x):
        return self.mlp(x)
class VAEFromTutorial(nn.Module):
    def __init__(self, shape, nhid = 16):
        super().__init__()
        self.dim = nhid
        self.encoder = Encoder(shape, nhid)
        self.decoder = Decoder(shape, nhid)
        
    def sampling(self, mean, logvar):
        eps = torch.randn(mean.shape, device=mean.device)
        sigma = torch.exp(0.5 * logvar)
        return mean + eps * sigma
    
    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.sampling(mean, logvar)
        return self.decoder(z), mean, logvar
    
    def decode(self, z, batch_size = None):
        res = self.decoder(z)
        if not batch_size:
            res = res.squeeze(0)
        return res

    def generate(self, batch_size = None):
        z = torch.randn((batch_size, self.dim)) if batch_size else torch.randn((1, self.dim))
        res = self.decoder(z)
        if not batch_size:
            res = res.squeeze(0)
        return res
    
    def to_gpus(self):
        self.cuda()

class FullyConnectedVAE2(nn.Module):
    def __init__(self, shape, num_classes:int, conditional_dim:int) -> None:
        super().__init__()
        #for now:
        assert len(shape) == 3  
        # shape: (C, H, W)
        self.label_embedding = lambda y : y
    
        num_dimensions = np.prod(shape)
        self.num_dimensions=num_dimensions
        self.encoder = nn.Sequential(
                                nn.Linear(num_dimensions, num_dimensions),
                                nn.BatchNorm1d(num_dimensions),
                                nn.LeakyReLU(),
                                nn.Linear(num_dimensions,num_dimensions),
                                nn.BatchNorm1d(num_dimensions),
                                nn.LeakyReLU())
        
        self.compute_mean = nn.Linear(num_dimensions, num_dimensions)
        self.compute_log_var = nn.Linear(num_dimensions, num_dimensions)
        
        self.decoder = nn.Sequential(nn.Linear(num_dimensions, num_dimensions),
                                nn.BatchNorm1d(num_dimensions),
                                nn.LeakyReLU(),
                                nn.Linear(num_dimensions, num_dimensions),
                                nn.BatchNorm1d(num_dimensions),
                                nn.LeakyReLU(),
                                nn.Linear(num_dimensions,num_dimensions),
                                nn.Unflatten(1, shape),
                                nn.LeakyReLU()
                                )
    
    def sample_from(self, mean, log_var):
        eps = torch.randn(mean.shape, device=mean.device)
        var =  torch.exp(0.5 *log_var)
        return mean + eps * var

    def encode(self, x, y=None):
        x = torch.flatten(x, start_dim=1)
        x = self.encoder(x)
        return self.compute_mean(x), self.compute_log_var(x)

    def decode(self, z, y=None):
        return self.decoder(z)

    def forward(self, x, y=None, z=None):
        mean, log_var = self.encode(x)

        z = self.sample_from(mean, log_var)
        x_hat = self.decode(z)
        return x_hat, mean, log_var

    def generate(self, y=None, z=None, batch_size: int=1):
        z = torch.randn((batch_size, self.num_dimensions)).cuda()
        return self.decode(z)
    
    def to_gpus(self):
        self.cuda(0)
        

class CFullyConnectedVAE(nn.Module):
    def __init__(self, shape, num_classes:int, conditional_dim:int) -> None:
        super().__init__()
        #for now:
        assert len(shape) == 3  
        # shape: (C, H, W)
        self.label_embedding = nn.Embedding(num_classes, conditional_dim)
    
        num_dimensions = np.prod(shape)
        self.num_dimensions=num_dimensions
        self.encoder = nn.Sequential(
                                nn.Linear(num_dimensions + conditional_dim, num_dimensions),
                                nn.BatchNorm1d(num_dimensions),
                                nn.LeakyReLU(),
                                nn.Linear(num_dimensions,num_dimensions),
                                nn.BatchNorm1d(num_dimensions),
                                nn.LeakyReLU())
        
        self.compute_mean = nn.Linear(num_dimensions, num_dimensions)
        self.compute_log_var = nn.Linear(num_dimensions, num_dimensions)
        
        self.decoder = nn.Sequential(nn.Linear(num_dimensions + conditional_dim, num_dimensions),
                                nn.BatchNorm1d(num_dimensions),
                                nn.LeakyReLU(),
                                     nn.Linear(num_dimensions,num_dimensions),
                                     nn.Unflatten(1, shape),
                                     #nn.Sigmoid()
                                )
    
    def sample_from(self, mean, log_var):
        eps = torch.randn(mean.shape, device=mean.device)
        var =  torch.exp(0.5 *log_var)
        return mean + eps * var

    def encode(self, x, y):
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x,y), dim=1)
        x = self.encoder(x)
        return self.compute_mean(x), self.compute_log_var(x)

    def decode(self, z, y):
        z = torch.cat((z,y), dim=1)
        return self.decoder(z)

    def forward(self, x, y, z=None):
        y = self.label_embedding(y)
        mean, log_var = self.encode(x , y)

        z = self.sample_from(mean, log_var)
        x_hat = self.decode(z, y)
        return x_hat, mean, log_var

    def generate(self, y, z=None, batch_size: int=1):
        y = self.label_embedding(y)
        z = torch.randn((batch_size, self.num_dimensions)).cuda()
        return self.decode(z, y)
    
    def to_gpus(self):
        self.cuda(0)

class CFullyConnectedVAE2(nn.Module):
    def __init__(self, shape, num_classes:int, conditional_dim:int) -> None:
        super().__init__()
        #for now:
        assert len(shape) == 3  
        # shape: (C, H, W)
        self.label_embedding = nn.Embedding(num_classes, conditional_dim)
    
        num_dimensions = np.prod(shape)
        self.num_dimensions=num_dimensions
        self.encoder = nn.Sequential(
                                nn.Linear(num_dimensions + conditional_dim, num_dimensions),
                                nn.BatchNorm1d(num_dimensions),
                                nn.LeakyReLU(),
                                nn.Linear(num_dimensions,num_dimensions),
                                nn.BatchNorm1d(num_dimensions),
                                nn.LeakyReLU())
        
        self.compute_mean = nn.Linear(num_dimensions, num_dimensions)
        self.compute_log_var = nn.Linear(num_dimensions, num_dimensions)
        
        self.decoder = nn.Sequential(nn.Linear(num_dimensions + conditional_dim, num_dimensions),
                                nn.BatchNorm1d(num_dimensions),
                                nn.LeakyReLU(),
                                nn.Linear(num_dimensions, num_dimensions),
                                nn.BatchNorm1d(num_dimensions),
                                nn.LeakyReLU(),
                                nn.Linear(num_dimensions,num_dimensions),
                                nn.Unflatten(1, shape),
                                nn.LeakyReLU()
                                )
    
    def sample_from(self, mean, log_var):
        if not self.training:
            return mean
        eps = torch.randn(mean.shape, device=mean.device)
        var =  torch.exp(0.5 *log_var)
        return mean + eps * var

    def encode(self, x, y):
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x,y), dim=1)
        x = self.encoder(x)
        return self.compute_mean(x), self.compute_log_var(x)

    def decode(self, z, y):
        z = torch.cat((z,y), dim=1)
        return self.decoder(z)

    def forward(self, x, y, slice_idx_normalized=None):
        y = self.label_embedding(y)
        mean, log_var = self.encode(x , y)

        z = self.sample_from(mean, log_var)
        x_hat = self.decode(z, y)
        return x_hat, mean, log_var

    def compute_z(self, x, y, slice_idx_normalized=None):
        y = self.label_embedding(y)
        mean, log_var = self.encode(x , y)
        z = self.sample_from(mean, log_var)
        return z


    def generate(self, y, slice_idx=None, batch_size: int=1):
        y = self.label_embedding(y)
        z = torch.randn((batch_size, self.num_dimensions)).cuda()
        return self.decode(z, y)
    
    def to_gpus(self):
        self.cuda(0)

class CFullyConnectedVAE4(nn.Module):
    def __init__(self, shape, num_classes:int, conditional_dim:int) -> None:
        super().__init__()
        #for now:
        assert len(shape) == 3  
        # shape: (C, H, W)
        self.label_embedding = nn.Embedding(num_classes, conditional_dim)
    
        num_dimensions = np.prod(shape)
        self.num_dimensions=num_dimensions
        self.encoder = nn.Sequential(
                                nn.Linear(num_dimensions + conditional_dim, num_dimensions + conditional_dim),
                                nn.BatchNorm1d(num_dimensions + conditional_dim),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(num_dimensions + conditional_dim, int(1.5 * num_dimensions)),
                                nn.BatchNorm1d(int(1.5 * num_dimensions)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(int(1.5 * num_dimensions), int(1.2 * num_dimensions)),
                                nn.BatchNorm1d(int(1.2 * num_dimensions)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(int(1.2 * num_dimensions),int(1.2 * num_dimensions)),
                                nn.BatchNorm1d(int(1.2 * num_dimensions)),
                                nn.LeakyReLU(inplace=True))
        
        self.compute_mean = nn.Linear(int(1.2 * num_dimensions), num_dimensions)
        self.compute_log_var = nn.Linear(int(1.2 * num_dimensions), num_dimensions)
        
        self.decoder = nn.Sequential(nn.Linear(num_dimensions + conditional_dim, num_dimensions + conditional_dim),
                                nn.BatchNorm1d(num_dimensions + conditional_dim),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(num_dimensions + conditional_dim, int(1.5 * num_dimensions)),
                                nn.BatchNorm1d(int(1.5 * num_dimensions)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(int(1.5 * num_dimensions), int(1.2 * num_dimensions)),
                                nn.BatchNorm1d(int(1.2 * num_dimensions)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(int(1.2 * num_dimensions), int(1.2 * num_dimensions)),
                                nn.BatchNorm1d(int(1.2 * num_dimensions)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(int(1.2 * num_dimensions), num_dimensions),
                                nn.Unflatten(1, shape),
                                nn.LeakyReLU(inplace=True)
                                )
    
    def sample_from(self, mean, log_var):
        if not self.training:
            return mean
        eps = torch.randn(mean.shape, device=mean.device)
        var =  torch.exp(0.5 *log_var)
        return mean + eps * var

    def encode(self, x, y):
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x,y), dim=1)
        x = self.encoder(x)
        return self.compute_mean(x), self.compute_log_var(x)

    def decode(self, z, y):
        z = torch.cat((z,y), dim=1)
        return self.decoder(z)

    def forward(self, x, y, slice_idx_normalized=None):
        y = self.label_embedding(y)
        mean, log_var = self.encode(x , y)

        z = self.sample_from(mean, log_var)
        x_hat = self.decode(z, y)
        return x_hat, mean, log_var

    def generate(self, y, slice_idx=None, batch_size: int=1):
        y = self.label_embedding(y)
        z = torch.randn((batch_size, self.num_dimensions)).cuda()
        return self.decode(z, y)
    
    def to_gpus(self):
        self.cuda(0)


class CFullyConnectedVAE4Distributed(nn.Module):
    def __init__(self, shape, num_classes:int, conditional_dim:int) -> None:
        super().__init__()
        #for now:
        assert len(shape) == 3  
        # shape: (C, H, W)
        self.label_embedding = nn.Embedding(num_classes, conditional_dim)
    
        num_dimensions = np.prod(shape)
        self.num_dimensions=num_dimensions
        self.encoder = nn.Sequential(
                                nn.Linear(num_dimensions + conditional_dim, num_dimensions + int(0.2 * conditional_dim), bias=False),
                                nn.BatchNorm1d(num_dimensions + int(0.2 * conditional_dim)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(num_dimensions + int(0.2 * conditional_dim), int(1.0 * num_dimensions), bias=False),
                                nn.BatchNorm1d(int(1.0 * num_dimensions)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(int(1.0 * num_dimensions), int(1.0 * num_dimensions), bias=False),
                                nn.BatchNorm1d(int(1.0 * num_dimensions)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(int(1.0 * num_dimensions),int(1.0 * num_dimensions), bias=False),
                                nn.BatchNorm1d(int(1.0 * num_dimensions)),
                                nn.LeakyReLU(inplace=True)
                                )
        
        self.compute_mean = nn.Linear(int(1.0 * num_dimensions), num_dimensions)
        self.compute_log_var = nn.Linear(int(1.0 * num_dimensions), num_dimensions)
        
        self.decoder = nn.Sequential(nn.Linear(num_dimensions + conditional_dim, num_dimensions + int(0.2 * conditional_dim), bias=False),
                                nn.BatchNorm1d(num_dimensions + int(0.2 * conditional_dim)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(num_dimensions + int(0.2 * conditional_dim), int(1.0 * num_dimensions), bias=False),
                                nn.BatchNorm1d(int(1.0 * num_dimensions)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(int(1.0 * num_dimensions), int(1.0 * num_dimensions), bias=False),
                                nn.BatchNorm1d(int(1.0 * num_dimensions)),
                                nn.LeakyReLU(inplace=True),
                                #nn.Linear(int(1.0 * num_dimensions), int(1.0 * num_dimensions), bias=False),
                                #nn.BatchNorm1d(int(1.0 * num_dimensions)),
                                #nn.LeakyReLU(inplace=True),
                                nn.Linear(int(1.0 * num_dimensions), num_dimensions),
                                nn.LeakyReLU(inplace=True),
                                nn.Unflatten(1, shape)
                                )
    
    def sample_from(self, mean, log_var):
        if not self.training:
            return mean
        eps = torch.randn(mean.shape, device=mean.device)
        var =  torch.exp(0.5 *log_var)
        return mean + eps * var

    def encode(self, x, y):
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x,y), dim=1)
        x = self.encoder(x)
        return self.compute_mean(x.cuda(1)), self.compute_log_var(x)

    def decode(self, z, y):
        z = torch.cat((z,y), dim=1)
        return self.decoder(z)

    def forward(self, x, y, slice_idx_normalized=None):
        y = self.label_embedding(y)
        mean, log_var = self.encode(x , y)

        z = self.sample_from(mean, log_var.cuda(1))
        x_hat = self.decode(z, y.cuda(1))
        return x_hat.cuda(0), mean.cuda(0), log_var

    def generate(self, y, slice_idx=None, batch_size: int=1):
        y = self.label_embedding(y).cuda(1)
        z = torch.randn((batch_size, self.num_dimensions), device='cuda:1')
        return self.decode(z, y)
    
    def to_gpus(self):
        self.label_embedding.cuda(0)
        self.encoder.cuda(0)
        self.compute_log_var.cuda(0)
        self.compute_mean.cuda(1)

        self.decoder.cuda(1)





class CFullyConnectedVAE4ConditionOnSlice(nn.Module):
    def __init__(self, shape, num_classes:int, conditional_dim:int) -> None:
        super().__init__()
        #for now:
        assert len(shape) == 3  
        # shape: (C, H, W)
        self.slice_embedding = nn.Embedding(10, conditional_dim)
    
        num_dimensions = np.prod(shape)
        self.num_dimensions=num_dimensions
        self.encoder = nn.Sequential(
                                nn.Linear(num_dimensions + conditional_dim, num_dimensions + conditional_dim),
                                nn.BatchNorm1d(num_dimensions + conditional_dim),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(num_dimensions + conditional_dim, int(1.5 * num_dimensions)),
                                nn.BatchNorm1d(int(1.5 * num_dimensions)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(int(1.5 * num_dimensions), int(1.2 * num_dimensions)),
                                nn.BatchNorm1d(int(1.2 * num_dimensions)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(int(1.2 * num_dimensions),int(1.2 * num_dimensions)),
                                nn.BatchNorm1d(int(1.2 * num_dimensions)),
                                nn.LeakyReLU(inplace=True))
        
        self.compute_mean = nn.Linear(int(1.2 * num_dimensions), num_dimensions)
        self.compute_log_var = nn.Linear(int(1.2 * num_dimensions), num_dimensions)
        
        self.decoder = nn.Sequential(nn.Linear(num_dimensions + conditional_dim, num_dimensions + conditional_dim),
                                nn.BatchNorm1d(num_dimensions + conditional_dim),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(num_dimensions + conditional_dim, int(1.5 * num_dimensions)),
                                nn.BatchNorm1d(int(1.5 * num_dimensions)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(int(1.5 * num_dimensions), int(1.2 * num_dimensions)),
                                nn.BatchNorm1d(int(1.2 * num_dimensions)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(int(1.2 * num_dimensions), int(1.2 * num_dimensions)),
                                nn.BatchNorm1d(int(1.2 * num_dimensions)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(int(1.2 * num_dimensions), num_dimensions),
                                nn.Unflatten(1, shape),
                                nn.LeakyReLU(inplace=True)
                                )
    
    def sample_from(self, mean, log_var):
        eps = torch.randn(mean.shape, device=mean.device)
        var =  torch.exp(0.5 *log_var)
        return mean + eps * var

    def encode(self, x, slice_idx):
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x,slice_idx), dim=1)
        x = self.encoder(x)
        return self.compute_mean(x), self.compute_log_var(x)

    def decode(self, z, slice_idx):
        z = torch.cat((z,slice_idx), dim=1)
        return self.decoder(z)

    def forward(self, x, y, slice_idx_normalized=None):
        slice_idx = torch.round(slice_idx_normalized * 9).long()
        slice_idx = self.slice_embedding(slice_idx)
        mean, log_var = self.encode(x, slice_idx)

        z = self.sample_from(mean, log_var)
        x_hat = self.decode(z, slice_idx)
        return x_hat, mean, log_var

    def generate(self, y, slice_idx=None, batch_size: int=1):
        slice_idx = torch.tensor([slice_idx], device='cuda:0')
        slice_idx = self.slice_embedding(slice_idx)
        z = torch.randn((batch_size, self.num_dimensions)).cuda()
        return self.decode(z, slice_idx)
    
    def to_gpus(self):
        self.cuda(0)

class CFullyConnectedVAE4ConditionOnBoth(nn.Module):
    def __init__(self, shape, num_classes:int, conditional_dim:int) -> None:
        super().__init__()
        #for now:
        assert len(shape) == 3  
        # shape: (C, H, W)
        self.label_embedding = nn.Embedding(num_classes, conditional_dim //2)
        self.slice_embedding = nn.Embedding(10, conditional_dim - conditional_dim //2)
    
        num_dimensions = np.prod(shape)
        self.num_dimensions=num_dimensions
        self.encoder = nn.Sequential(
                                nn.Linear(num_dimensions + conditional_dim, num_dimensions + conditional_dim),
                                nn.BatchNorm1d(num_dimensions + conditional_dim),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(num_dimensions + conditional_dim, int(1.5 * num_dimensions)),
                                nn.BatchNorm1d(int(1.5 * num_dimensions)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(int(1.5 * num_dimensions), int(1.2 * num_dimensions)),
                                nn.BatchNorm1d(int(1.2 * num_dimensions)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(int(1.2 * num_dimensions),int(1.2 * num_dimensions)),
                                nn.BatchNorm1d(int(1.2 * num_dimensions)),
                                nn.LeakyReLU(inplace=True))
        
        self.compute_mean = nn.Linear(int(1.2 * num_dimensions), num_dimensions)
        self.compute_log_var = nn.Linear(int(1.2 * num_dimensions), num_dimensions)
        
        self.decoder = nn.Sequential(nn.Linear(num_dimensions + conditional_dim, num_dimensions + conditional_dim),
                                nn.BatchNorm1d(num_dimensions + conditional_dim),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(num_dimensions + conditional_dim, int(1.5 * num_dimensions)),
                                nn.BatchNorm1d(int(1.5 * num_dimensions)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(int(1.5 * num_dimensions), int(1.2 * num_dimensions)),
                                nn.BatchNorm1d(int(1.2 * num_dimensions)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(int(1.2 * num_dimensions), int(1.2 * num_dimensions)),
                                nn.BatchNorm1d(int(1.2 * num_dimensions)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(int(1.2 * num_dimensions), num_dimensions),
                                nn.Unflatten(1, shape),
                                nn.LeakyReLU(inplace=True)
                                )
    
    def sample_from(self, mean, log_var):
        if not self.training:
            return mean
        eps = torch.randn(mean.shape, device=mean.device)
        var =  torch.exp(0.5 *log_var)
        return mean + eps * var

    def encode(self, x, y, slice_idx):
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x, y, slice_idx), dim=1)
        x = self.encoder(x)
        return self.compute_mean(x), self.compute_log_var(x)

    def decode(self, z, y, slice_idx):
        z = torch.cat((z, y, slice_idx), dim=1)
        return self.decoder(z)

    def forward(self, x, y, slice_idx_normalized=None):
        y = self.label_embedding(y)
        slice_idx = torch.round(slice_idx_normalized * 9).long()
        slice_idx = self.slice_embedding(slice_idx)
        mean, log_var = self.encode(x, y, slice_idx)

        z = self.sample_from(mean, log_var)
        x_hat = self.decode(z, y, slice_idx)
        return x_hat, mean, log_var

    def generate(self, y, slice_idx, batch_size: int=1):
        y = self.label_embedding(y)
        slice_idx = torch.tensor([slice_idx], device='cuda:0')
        slice_idx = self.slice_embedding(slice_idx)
        z = torch.randn((batch_size, self.num_dimensions)).cuda()
        return self.decode(z, y, slice_idx)
    
    def compute_z(self, x, y, slice_idx_normalized=None):
        y = self.label_embedding(y)
        slice_idx = torch.round(slice_idx_normalized * 9).long()
        slice_idx = self.slice_embedding(slice_idx)
        mean, log_var = self.encode(x, y, slice_idx)
        z = self.sample_from(mean, log_var)
        return z
    
    def to_gpus(self):
        self.cuda(0)


class CFullyConnectedVAE4ConditionOnBothDistributed(nn.Module):
    def __init__(self, shape, num_classes:int, conditional_dim:int) -> None:
        super().__init__()
        #for now:
        assert len(shape) == 3  
        # shape: (C, H, W)
        self.label_embedding = nn.Embedding(num_classes, conditional_dim //2)
        self.slice_embedding = nn.Embedding(10, conditional_dim - conditional_dim //2)
    
        num_dimensions = np.prod(shape)
        self.num_dimensions=num_dimensions
        self.encoder = nn.Sequential(
                                nn.Linear(num_dimensions + conditional_dim, num_dimensions + int(0.2 * conditional_dim), bias=False),
                                nn.BatchNorm1d(num_dimensions + int(0.2 * conditional_dim)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(num_dimensions + int(0.2 * conditional_dim), int(1.0 * num_dimensions), bias=False),
                                nn.BatchNorm1d(int(1.0 * num_dimensions)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(int(1.0 * num_dimensions), int(1.0 * num_dimensions), bias=False),
                                nn.BatchNorm1d(int(1.0 * num_dimensions)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(int(1.0 * num_dimensions),int(1.0 * num_dimensions), bias=False),
                                nn.BatchNorm1d(int(1.0 * num_dimensions)),
                                nn.LeakyReLU(inplace=True)
                                )
        
        self.compute_mean = nn.Linear(int(1.0 * num_dimensions), num_dimensions)
        self.compute_log_var = nn.Linear(int(1.0 * num_dimensions), num_dimensions)
        
        self.decoder = nn.Sequential(nn.Linear(num_dimensions + conditional_dim, num_dimensions + int(0.2 * conditional_dim), bias=False),
                                nn.BatchNorm1d(num_dimensions + int(0.2 * conditional_dim)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(num_dimensions + int(0.2 * conditional_dim), int(1.0 * num_dimensions), bias=False),
                                nn.BatchNorm1d(int(1.0 * num_dimensions)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(int(1.0 * num_dimensions), int(1.0 * num_dimensions), bias=False),
                                nn.BatchNorm1d(int(1.0 * num_dimensions)),
                                nn.LeakyReLU(inplace=True),
                                #nn.Linear(int(1.0 * num_dimensions), int(1.0 * num_dimensions), bias=False),
                                #nn.BatchNorm1d(int(1.0 * num_dimensions)),
                                #nn.LeakyReLU(inplace=True),
                                nn.Linear(int(1.0 * num_dimensions), num_dimensions),
                                nn.LeakyReLU(inplace=True),
                                nn.Unflatten(1, shape)
                                )
    
    def sample_from(self, mean, log_var):
        if not self.training:
            return mean
        eps = torch.randn(mean.shape, device=mean.device)
        var =  torch.exp(0.5 *log_var)
        return mean + eps * var

    def encode(self, x, y, slice_idx):
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x, y, slice_idx), dim=1)
        x = self.encoder(x)
        
        return self.compute_mean(x.cuda(1)), self.compute_log_var(x).cuda(1)

    def decode(self, z, y, slice_idx):
        z = torch.cat((z, y, slice_idx), dim=1)
        return self.decoder(z)

    def forward(self, x, y, slice_idx_normalized=None):
        y = self.label_embedding(y)
        slice_idx = torch.round(slice_idx_normalized * 9).long()
        slice_idx = self.slice_embedding(slice_idx)
        mean, log_var = self.encode(x, y, slice_idx)

        z = self.sample_from(mean, log_var)
        x_hat = self.decode(z, y.cuda(1), slice_idx.cuda(1))
        return x_hat.cuda(0), mean.cuda(0), log_var.cuda(0)

    def generate(self, y, slice_idx, batch_size: int=1):
        y = self.label_embedding(y).cuda(1)
        slice_idx = torch.tensor([slice_idx], device='cuda:0')
        slice_idx = self.slice_embedding(slice_idx).cuda(1)
        z = torch.randn((batch_size, self.num_dimensions), device='cuda:1')
        return self.decode(z, y, slice_idx)
    
    def compute_z(self, x, y, slice_idx_normalized=None):
        y = self.label_embedding(y)
        slice_idx = torch.round(slice_idx_normalized * 9).long()
        slice_idx = self.slice_embedding(slice_idx)
        mean, log_var = self.encode(x, y, slice_idx)
        z = self.sample_from(mean, log_var)
        return z
    
    def to_gpus(self):
        self.label_embedding.cuda(0)
        self.slice_embedding.cuda(0)
        self.encoder.cuda(0)

        self.compute_log_var.cuda(0)
        self.compute_mean.cuda(1)
        self.decoder.cuda(1)



class FullyConnectedVAE2Distributed(nn.Module):
    def __init__(self, shape, num_classes:int, conditional_dim:int) -> None:
        super().__init__()
        #for now:
        assert len(shape) == 3  
        # shape: (C, H, W)
        self.label_embedding = lambda y : y

        num_dimensions = np.prod(shape)
        self.num_dimensions=num_dimensions
        self.encoder = nn.Sequential(
                                nn.Linear(num_dimensions, num_dimensions),
                                nn.BatchNorm1d(num_dimensions),
                                nn.LeakyReLU(),
                                nn.Linear(num_dimensions,num_dimensions),
                                nn.BatchNorm1d(num_dimensions),
                                nn.LeakyReLU())
        
        self.compute_mean = nn.Linear(num_dimensions, num_dimensions)
        self.compute_log_var = nn.Linear(num_dimensions, num_dimensions)
        
        self.decoder = nn.Sequential(nn.Linear(num_dimensions, num_dimensions),
                                nn.BatchNorm1d(num_dimensions),
                                nn.LeakyReLU(),
                                nn.Linear(num_dimensions, num_dimensions),
                                nn.BatchNorm1d(num_dimensions),
                                nn.LeakyReLU(),
                                nn.Linear(num_dimensions,num_dimensions),
                                nn.Unflatten(1, shape),
                                nn.LeakyReLU()
                                )
    
    def sample_from(self, mean, log_var):
        eps = torch.randn(mean.shape, device=mean.device)
        var =  torch.exp(0.5 *log_var)
        return mean + eps * var

    def encode(self, x, y=None):
        x = torch.flatten(x, start_dim=1)
        x = self.encoder(x)
        return self.compute_mean(x), self.compute_log_var(x)

    def decode(self, z, y=None):
        return self.decoder(z)

    def forward(self, x, y=None, z=None):
        mean, log_var = self.encode(x)

        z = self.sample_from(mean, log_var).cuda(1)
        x_hat = self.decode(z)
        return x_hat.cuda(0), mean, log_var

    def generate(self, y=None, z=None, batch_size: int=1):
        z = torch.randn((batch_size, self.num_dimensions)).cuda(1)
        return self.decode(z, y)
    
    def to_gpus(self):
        self.label_embedding.cuda(0)
        self.encoder.cuda(0)
        self.compute_log_var.cuda(0)
        self.compute_mean.cuda(0)

        self.decoder.cuda(1)

class CFullyConnectedVAE2Distributed(nn.Module):
    def __init__(self, shape, num_classes:int, conditional_dim:int) -> None:
        super().__init__()
        #for now:
        assert len(shape) == 3  
        # shape: (C, H, W)
        self.label_embedding = nn.Embedding(num_classes, conditional_dim)
    
        num_dimensions = np.prod(shape)
        self.num_dimensions=num_dimensions
        self.encoder = nn.Sequential(
                                nn.Linear(num_dimensions + conditional_dim, num_dimensions),
                                nn.BatchNorm1d(num_dimensions),
                                nn.LeakyReLU(),
                                nn.Linear(num_dimensions,num_dimensions),
                                nn.BatchNorm1d(num_dimensions),
                                nn.LeakyReLU())
        
        self.compute_mean = nn.Linear(num_dimensions, num_dimensions)
        self.compute_log_var = nn.Linear(num_dimensions, num_dimensions)
        
        self.decoder = nn.Sequential(nn.Linear(num_dimensions + conditional_dim, num_dimensions),
                                nn.BatchNorm1d(num_dimensions),
                                nn.LeakyReLU(),
                                nn.Linear(num_dimensions, num_dimensions),
                                nn.BatchNorm1d(num_dimensions),
                                nn.LeakyReLU(),
                                nn.Linear(num_dimensions,num_dimensions),
                                nn.Unflatten(1, shape),
                                nn.LeakyReLU()
                                )
    
    def sample_from(self, mean, log_var):
        if not self.training:
            return mean
        eps = torch.randn(mean.shape, device=mean.device)
        var =  torch.exp(0.5 *log_var)
        return mean + eps * var

    def encode(self, x, y):
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x,y), dim=1)
        x = self.encoder(x)
        return self.compute_mean(x), self.compute_log_var(x)

    def decode(self, z, y):
        z = torch.cat((z,y), dim=1)
        return self.decoder(z)

    def forward(self, x, y, slice_idx_normalized=None):
        y = self.label_embedding(y)
        mean, log_var = self.encode(x , y)

        z = self.sample_from(mean, log_var).cuda(1)
        x_hat = self.decode(z, y.cuda(1))
        return x_hat.cuda(0), mean, log_var

    def generate(self, y, slice_idx=None, batch_size: int=1):
        y = self.label_embedding(y).cuda(1)
        z = torch.randn((batch_size, self.num_dimensions)).cuda(1)
        return self.decode(z, y)
    
    def to_gpus(self):
        self.label_embedding.cuda(0)
        self.encoder.cuda(0)
        self.compute_log_var.cuda(0)
        self.compute_mean.cuda(0)

        self.decoder.cuda(1)


class CFullyConnectedVAE3(nn.Module):
    def __init__(self, shape, num_classes:int, conditional_dim:int) -> None:
        super().__init__()
        #for now:
        assert len(shape) == 3  
        # shape: (C, H, W)
        self.label_embedding = nn.Embedding(num_classes, conditional_dim)
    
        num_dimensions = np.prod(shape)
        self.num_dimensions=num_dimensions
        self.encoder = nn.Sequential(
                                nn.Linear(num_dimensions + conditional_dim, num_dimensions),
                                nn.BatchNorm1d(num_dimensions),
                                nn.LeakyReLU(),
                                nn.Linear(num_dimensions,num_dimensions),
                                nn.BatchNorm1d(num_dimensions),
                                nn.LeakyReLU(),
                                nn.Linear(num_dimensions,num_dimensions),
                                nn.BatchNorm1d(num_dimensions),
                                nn.LeakyReLU())
        
        self.compute_mean = nn.Linear(num_dimensions, num_dimensions)
        self.compute_log_var = nn.Linear(num_dimensions, num_dimensions)
        
        self.decoder = nn.Sequential(nn.Linear(num_dimensions + conditional_dim, num_dimensions),
                                nn.BatchNorm1d(num_dimensions),
                                nn.LeakyReLU(),
                                nn.Linear(num_dimensions, num_dimensions),
                                nn.BatchNorm1d(num_dimensions),
                                nn.LeakyReLU(),
                                nn.Linear(num_dimensions, num_dimensions),
                                nn.BatchNorm1d(num_dimensions),
                                nn.LeakyReLU(),
                                nn.Linear(num_dimensions,num_dimensions),
                                nn.Unflatten(1, shape),
                                nn.LeakyReLU()
                                )
    
    def sample_from(self, mean, log_var):
        eps = torch.randn(mean.shape, device=mean.device)
        var =  torch.exp(0.5 *log_var)
        return mean + eps * var

    def encode(self, x, y):
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x,y), dim=1)
        x = self.encoder(x)
        return self.compute_mean(x), self.compute_log_var(x.cuda(1))

    def decode(self, z, y):
        z = torch.cat((z.cuda(1), y.cuda(1)), dim=1)
        return self.decoder(z)

    def forward(self, x, y, z=None):
        y = self.label_embedding(y)
        mean, log_var = self.encode(x , y)

        mean_1 = mean.cuda(1)
        z = self.sample_from(mean_1, log_var)
        x_hat = self.decode(z, y)
        return x_hat.cuda(0), mean, log_var.cuda(0)

    def generate(self, y, z=None, batch_size: int=1):
        y = y.cuda(0)
        y = self.label_embedding(y).cuda(1)
        z = torch.randn((batch_size, self.num_dimensions)).cuda(1)
        return self.decode(z, y)
    
    def to_gpus(self):
        self.label_embedding.cuda(0)
        self.encoder.cuda(0)
        self.compute_mean.cuda(0)
        self.compute_log_var.cuda(1)
        self.decoder.cuda(1)


class FullyConnectedVAE(nn.Module):
    def __init__(self, shape) -> None:
        super().__init__()
        #for now:
        assert len(shape) == 3  
        # shape: (C, H, W)
    
        num_dimensions = np.prod(shape)
        self.num_dimensions=num_dimensions

        """
        self.encoder = nn.Sequential(nn.Flatten(),
                                nn.Linear(num_dimensions,10000),
                                nn.BatchNorm1d(10000),
                                nn.LeakyReLU())
        
        self.compute_mean = nn.Linear(10000, num_dimensions)
        self.compute_log_var = nn.Linear(10000, num_dimensions)
        self.decoder = nn.Sequential(nn.Linear(num_dimensions,10000),
                                nn.BatchNorm1d(10000),
                                nn.LeakyReLU(),
                                     nn.Linear(10000,num_dimensions),
                                     nn.Unflatten(1, shape))
        """
        
        self.encoder = nn.Sequential(nn.Flatten(),
                                nn.Linear(num_dimensions,10000),
                                nn.ReLU(),
                                nn.Linear(10000,5000),
                                nn.ReLU())
        
        self.compute_mean = nn.Linear(5000 + num_dimensions, num_dimensions)
        self.compute_log_var = nn.Linear(5000 + num_dimensions, num_dimensions)
        
        self.decoder = nn.Sequential(nn.Linear(num_dimensions,10000),
                                nn.ReLU(),
                                nn.Linear(10000,5000),
                                nn.ReLU())
        
        self.final_layers = nn.Sequential(
                                     nn.Linear(5000 + num_dimensions, num_dimensions),
                                     nn.Unflatten(1, shape))
    
        """
        self.encoder = nn.Sequential(nn.Flatten(),
                                nn.Linear(num_dimensions,10000),
                                nn.BatchNorm1d(10000),
                                nn.LeakyReLU(),
                                nn.Linear(10000,10000),
                                nn.BatchNorm1d(10000),
                                nn.LeakyReLU())
        
        self.compute_mean = nn.Linear(10000, num_dimensions-1000)
        self.compute_log_var = nn.Linear(10000, num_dimensions-1000)
        
        self.decoder = nn.Sequential(nn.Linear(num_dimensions-1000,15000),
                                nn.BatchNorm1d(15000),
                                nn.LeakyReLU(),
                                     nn.Linear(15000,num_dimensions),
                                     nn.Unflatten(1, shape))
        """
        """
        self.encoder = nn.Sequential(nn.Flatten(),
                                nn.Linear(num_dimensions,10000),
                                nn.BatchNorm1d(10000),
                                nn.LeakyReLU(),
                                nn.Linear(10000,15000),
                                nn.BatchNorm1d(15000),
                                nn.LeakyReLU())
        
        self.compute_mean = nn.Linear(15000, num_dimensions)
        self.compute_log_var = nn.Linear(15000, num_dimensions)
        self.decoder = nn.Sequential(nn.Linear(num_dimensions,10000),
                                nn.BatchNorm1d(10000),
                                nn.LeakyReLU(),
                                nn.Linear(10000,10000),
                                nn.BatchNorm1d(10000),
                                nn.LeakyReLU(),
                                nn.Linear(10000,10000),
                                nn.BatchNorm1d(10000),
                                nn.LeakyReLU(),
                                nn.Linear(10000,num_dimensions),
                                nn.Unflatten(1, shape))
        """
        """
        self.encoder = nn.Sequential(
                                # 256, 7, 5 = 8,960
                                nn.Conv2d(256,512,3,1,1),
                                # 512, 7, 5 = 17,920
                                nn.BatchNorm2d(512),
                                nn.LeakyReLU(),
                                nn.Conv2d(512,1024,3,1,1),
                                # 1024, 7, 5 = 35,840
                                nn.BatchNorm2d(1024),
                                nn.LeakyReLU(),
                                nn.Conv2d(1024,2048,3,2,1),
                                # 2048, 4, 3 = 24,576
                                nn.BatchNorm2d(2048),
                                nn.LeakyReLU(),
                                nn.Conv2d(2048,2048,3,1,1),
                                # 2048, 4, 3 = 24,576
                                nn.BatchNorm2d(2048),
                                nn.LeakyReLU(),
                                nn.Conv2d(2048,2500,3,2,1),
                                # 2500, 2, 2 = 10,000
                                nn.BatchNorm2d(2500),
                                nn.LeakyReLU(),
                                nn.Flatten()
                                )
        
        self.compute_mean = nn.Linear(10000, 8960)
        self.compute_log_var = nn.Linear(10000, 8960)

        self.decoder = nn.Sequential(
                        nn.Linear(8960,10000),
                        # 10,000
                        nn.BatchNorm1d(10000),
                        nn.LeakyReLU(),
                                nn.Linear(num_dimensions,num_dimensions),
                                nn.Unflatten(1, shape))
        """
    def sample_from(self, mean, log_var):
        eps = torch.randn(mean.shape, device=mean.device)
        var =  torch.exp(0.5 *log_var)
        return mean + eps * var

    def encode(self, x):
        y = self.encoder(x)
        x = torch.flatten(x, 1, -1)
        x = torch.cat([x,y], dim=-1)
        return self.compute_mean(x), self.compute_log_var(x)

    def decode(self, z):
        z = z.cuda(1)
        y = self.decoder(z)
        return self.final_layers(torch.cat([z,y], dim=-1)).cuda(0)

    def forward(self, x):
        mean, log_var = self.encode(x)

        z = self.sample_from(mean, log_var)
        x_hat = self.decode(z)
        return x_hat, mean, log_var

    def generate(self, z=None, batch_size: int=1):
        z = torch.randn((batch_size, self.num_dimensions))
        return self.decode(z)
    
    def to_gpus(self):
        self.cuda(0)
        self.decoder.cuda(1)
        self.final_layers.cuda(1)

class ConvolutionalVAE(nn.Module):
    def __init__(self, shape, hidden_dim) -> None:
        super().__init__()

        #for now:
        assert len(shape) == 3  
        # shape: (C, H, W)

        self.latent_dim = hidden_dim

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
        var = torch.exp(0.5 * log_var)
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

    def generate(self, z=None, batch_size: int=1):
        z = torch.randn((batch_size, self.latent_dim)).cuda()
        return self.decode(z)

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
        var = torch.exp(0.5 * log_var)
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
    

class CFullyConnectedVAE4NoConditioning(nn.Module):
    def __init__(self, shape, num_classes:int, conditional_dim:int) -> None:
        super().__init__()
        #for now:
        assert len(shape) == 3  
        # shape: (C, H, W)
    
        num_dimensions = np.prod(shape)
        self.num_dimensions=num_dimensions
        self.encoder = nn.Sequential(
                                nn.Linear(num_dimensions, num_dimensions + conditional_dim),
                                nn.BatchNorm1d(num_dimensions + conditional_dim),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(num_dimensions + conditional_dim, int(1.5 * num_dimensions)),
                                nn.BatchNorm1d(int(1.5 * num_dimensions)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(int(1.5 * num_dimensions), int(1.2 * num_dimensions)),
                                nn.BatchNorm1d(int(1.2 * num_dimensions)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(int(1.2 * num_dimensions),int(1.2 * num_dimensions)),
                                nn.BatchNorm1d(int(1.2 * num_dimensions)),
                                nn.LeakyReLU(inplace=True))
        
        self.compute_mean = nn.Linear(int(1.2 * num_dimensions), num_dimensions)
        self.compute_log_var = nn.Linear(int(1.2 * num_dimensions), num_dimensions)
        
        self.decoder = nn.Sequential(nn.Linear(num_dimensions, num_dimensions + conditional_dim),
                                nn.BatchNorm1d(num_dimensions + conditional_dim),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(num_dimensions + conditional_dim, int(1.5 * num_dimensions)),
                                nn.BatchNorm1d(int(1.5 * num_dimensions)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(int(1.5 * num_dimensions), int(1.2 * num_dimensions)),
                                nn.BatchNorm1d(int(1.2 * num_dimensions)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(int(1.2 * num_dimensions), int(1.2 * num_dimensions)),
                                nn.BatchNorm1d(int(1.2 * num_dimensions)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(int(1.2 * num_dimensions), num_dimensions),
                                nn.Unflatten(1, shape),
                                nn.LeakyReLU(inplace=True)
                                )
    
    def sample_from(self, mean, log_var):
        if not self.training:
            return mean
        eps = torch.randn(mean.shape, device=mean.device)
        var =  torch.exp(0.5 *log_var)
        return mean + eps * var

    def encode(self, x, y=None):
        x = torch.flatten(x, start_dim=1)
        x = self.encoder(x)
        return self.compute_mean(x), self.compute_log_var(x)

    def decode(self, z, y=None):
        return self.decoder(z)

    def forward(self, x, y, slice_idx_normalized=None):
        mean, log_var = self.encode(x)

        z = self.sample_from(mean, log_var)
        x_hat = self.decode(z)
        return x_hat, mean, log_var

    def generate(self, y=None, slice_idx=None, batch_size: int=1):
        z = torch.randn((batch_size, self.num_dimensions)).cuda()
        return self.decode(z)
    
    def to_gpus(self):
        self.cuda(0)


class CFullyConnectedVAE4NoConditioningDistributed(nn.Module):
    def __init__(self, shape, num_classes:int, conditional_dim:int) -> None:
        super().__init__()
        #for now:
        assert len(shape) == 3  
        # shape: (C, H, W)
    
        num_dimensions = np.prod(shape)
        self.num_dimensions=num_dimensions
        self.encoder = nn.Sequential(
                                nn.Linear(num_dimensions, num_dimensions + int(0.2 * conditional_dim), bias=False),
                                nn.BatchNorm1d(num_dimensions + int(0.2 * conditional_dim)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(num_dimensions + int(0.2 * conditional_dim), int(1.0 * num_dimensions), bias=False),
                                nn.BatchNorm1d(int(1.0 * num_dimensions)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(int(1.0 * num_dimensions), int(1.0 * num_dimensions), bias=False),
                                nn.BatchNorm1d(int(1.0 * num_dimensions)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(int(1.0 * num_dimensions),int(1.0 * num_dimensions), bias=False),
                                nn.BatchNorm1d(int(1.0 * num_dimensions)),
                                nn.LeakyReLU(inplace=True)
                                )
        
        self.compute_mean = nn.Linear(int(1.0 * num_dimensions), num_dimensions)
        self.compute_log_var = nn.Linear(int(1.0 * num_dimensions), num_dimensions)
        
        self.decoder = nn.Sequential(nn.Linear(num_dimensions, num_dimensions + int(0.2 * conditional_dim), bias=False),
                                nn.BatchNorm1d(num_dimensions + int(0.2 * conditional_dim)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(num_dimensions + int(0.2 * conditional_dim), int(1.0 * num_dimensions), bias=False),
                                nn.BatchNorm1d(int(1.0 * num_dimensions)),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(int(1.0 * num_dimensions), int(1.0 * num_dimensions), bias=False),
                                nn.BatchNorm1d(int(1.0 * num_dimensions)),
                                nn.LeakyReLU(inplace=True),
                                #nn.Linear(int(1.0 * num_dimensions), int(1.0 * num_dimensions), bias=False),
                                #nn.BatchNorm1d(int(1.0 * num_dimensions)),
                                #nn.LeakyReLU(inplace=True),
                                nn.Linear(int(1.0 * num_dimensions), num_dimensions),
                                nn.LeakyReLU(inplace=True),
                                nn.Unflatten(1, shape)
                                )
    
    def sample_from(self, mean, log_var):
        if not self.training:
            return mean
        eps = torch.randn(mean.shape, device=mean.device)
        var =  torch.exp(0.5 *log_var)
        return mean + eps * var

    def encode(self, x, y=None):
        x = torch.flatten(x, start_dim=1)
        x = self.encoder(x)
        return self.compute_mean(x.cuda(1)), self.compute_log_var(x)

    def decode(self, z, y=None):
        return self.decoder(z)

    def forward(self, x, y=None, slice_idx_normalized=None):
        mean, log_var = self.encode(x)

        z = self.sample_from(mean, log_var.cuda(1))
        x_hat = self.decode(z)
        return x_hat.cuda(0), mean.cuda(0), log_var

    def generate(self, y=None, slice_idx=None, batch_size: int=1):
        z = torch.randn((batch_size, self.num_dimensions), device='cuda:1')
        return self.decode(z)
    
    def to_gpus(self):
        self.encoder.cuda(0)
        self.compute_log_var.cuda(0)
        self.compute_mean.cuda(1)

        self.decoder.cuda(1)
    
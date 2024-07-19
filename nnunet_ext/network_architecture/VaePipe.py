from typing import Any, Iterable, Mapping, Optional
from fairscale.nn.pipe.pipe import Devices
import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
import numpy as np
import fairscale
from fairscale.nn.pipe.balance import balance_by_time
from torch.nn.modules import Sequential

def distribute_evenly(num_layers, num_devices):
    # Calculate the base value to evenly distribute
    base_value = num_layers // num_devices
    
    # Calculate the remainder for adjustment
    remainder = num_layers % num_devices
    
    # Create a list with base values
    result = [base_value] * num_devices
    
    # Adjust the list by distributing the remainder
    for i in range(remainder):
        result[i] += 1
    
    return result


class RenamePipe(fairscale.nn.Pipe):
    def __init__(
        self,
        module: nn.Sequential,
        balance: Optional[Iterable[int]] = None,
        *,
        devices: Optional[Devices] = None,
        chunks: int = 1,
        checkpoint: str = "checkpoint",
        deferred_batch_norm: bool = False,
    ) -> None:
        super().__init__(module, balance, 
                         devices=devices, 
                         chunks=chunks, 
                         checkpoint=checkpoint, 
                         deferred_batch_norm=deferred_batch_norm)
        
        rename_map = {}

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        print(state_dict.keys())
        print(self)
        exit()
        return super().load_state_dict(state_dict, strict)





class Reparameterization(nn.Module):
    def forward(self, mean, log_var):
        if not self.training:
            return mean
        eps = torch.randn(mean.shape, device=mean.device)
        var =  torch.exp(0.5 *log_var)
        return mean + eps * var


class CFullyConnectedVAE4ConditionOnBothFairScale(nn.Module):
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
        
        
        all_nvidia_gpus = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        num_nvidia_gpus = len(all_nvidia_gpus)


        balances_and_devices = [(1, 'cuda:0'), (2, 'cuda:1'), (3, 'cuda:2'), (3, 'cuda:3'), (3, 'cuda:4'), 
                                (3, 'cuda:5'), (2, 'cuda:6'), (5, 'cuda:7')]

        balance = [1,5,4,2]
        devices = ['cuda:1', 'cuda:2', 'cuda:3', 'cuda:4']
        #balance = [4,5,3]
        #devices = [self.embedding_device, all_nvidia_gpus[1], all_nvidia_gpus[2]]
        self.encoder = fairscale.nn.Pipe(self.encoder, checkpoint='never', devices=devices, 
                                         balance=balance)
        
        balance = [2,5,5]
        devices = ['cuda:5', 'cuda:6', 'cuda:7']
        #balance = [2,5,5]
        #devices = [self.reparameterization_device, all_nvidia_gpus[3], all_nvidia_gpus[4]]
        self.decoder = fairscale.nn.Pipe(self.decoder, checkpoint='never', devices=devices, 
                                         balance=balance)
        
        self.embedding_device = 'cuda:0'
        self.reparameterization_device = 'cuda:4'
        self.label_embedding.cuda(self.embedding_device)
        self.slice_embedding.cuda(self.embedding_device)
        self.compute_log_var.cuda(self.reparameterization_device)
        self.compute_mean.cuda(self.reparameterization_device)
    
    def sample_from(self, mean, log_var):
        if not self.training:
            return mean
        eps = torch.randn(mean.shape, device=mean.device)
        var =  torch.exp(0.5 *log_var)
        return mean + eps * var

    def encode(self, x, y, slice_idx):
        x = torch.flatten(x, start_dim=1)
        if x.device != self.embedding_device:
            x = x.to(self.embedding_device)
        x = torch.cat((x, y, slice_idx), dim=1)
        x = self.encoder(x.to(self.encoder.devices[0]))
    
        if x.device != self.reparameterization_device:
            x = x.to(self.reparameterization_device)

        return self.compute_mean(x), self.compute_log_var(x)

    def decode(self, z, y, slice_idx):
        if z.device != self.decoder.devices[0]:
            z = z.to(self.decoder.devices[0])
        z = torch.cat((z, y.to(self.decoder.devices[0]), slice_idx.to(self.decoder.devices[0])), dim=1)
        return self.decoder(z)

    def forward(self, x, y, slice_idx_normalized=None):
        y = self.label_embedding(y.to(self.embedding_device))
        slice_idx = torch.round(slice_idx_normalized * 9).long().to(self.embedding_device)
        slice_idx = self.slice_embedding(slice_idx)
        mean, log_var = self.encode(x, y, slice_idx)

        z = self.sample_from(mean, log_var)
        x_hat = self.decode(z, y, slice_idx)
        return x_hat.cuda(0), mean.cuda(0), log_var.cuda(0)

    def generate(self, y, slice_idx, batch_size: int=1):
        y = self.label_embedding(y)
        slice_idx = torch.tensor([slice_idx], device=self.embedding_device)
        slice_idx = self.slice_embedding(slice_idx)
        z = torch.randn((batch_size, self.num_dimensions), device=self.decoder.devices[0])
        return self.decode(z, y, slice_idx)
    
    def to_gpus(self):
        return self
        self.label_embedding.cuda(0)
        self.slice_embedding.cuda(0)
        self.encoder.cuda(0)

        self.compute_log_var.cuda(0)
        self.compute_mean.cuda(1)
        self.decoder.cuda(1)



class FairScaleVAEPlugInConditionOnBoth(nn.Module):
    def __init__(self, vae) -> None:
        super().__init__()
        self.label_embedding = vae.label_embedding
        self.slice_embedding = vae.slice_embedding
    
        self.num_dimensions=vae.num_dimensions
        self.encoder = vae.encoder
        
        self.compute_mean = vae.compute_mean
        self.compute_log_var = vae.compute_log_var
        
        self.decoder = vae.decoder
        
        
        all_nvidia_gpus = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
        num_nvidia_gpus = len(all_nvidia_gpus)


        balances_and_devices = [(6, 'cuda:0'), (6, 'cuda:1'), (6, 'cuda:2'), (9, 'cuda:3')]

        balance = [1,5,4,2]
        devices = ['cuda:1', 'cuda:2', 'cuda:3', 'cuda:4']
        #balance = [4,5,3]
        #devices = [self.embedding_device, all_nvidia_gpus[1], all_nvidia_gpus[2]]
        self.encoder = fairscale.nn.Pipe(self.encoder, checkpoint='never', devices=devices, 
                                         balance=balance)
        
        balance = [2,5,5]
        devices = ['cuda:5', 'cuda:6', 'cuda:7']
        #balance = [2,5,5]
        #devices = [self.reparameterization_device, all_nvidia_gpus[3], all_nvidia_gpus[4]]
        self.decoder = fairscale.nn.Pipe(self.decoder, checkpoint='never', devices=devices, 
                                         balance=balance)
        
        self.embedding_device = 'cuda:0'
        self.reparameterization_device = 'cuda:4'
        self.label_embedding.cuda(self.embedding_device)
        self.slice_embedding.cuda(self.embedding_device)
        self.compute_log_var.cuda(self.reparameterization_device)
        self.compute_mean.cuda(self.reparameterization_device)
    
    def sample_from(self, mean, log_var):
        if not self.training:
            return mean
        eps = torch.randn(mean.shape, device=mean.device)
        var =  torch.exp(0.5 *log_var)
        return mean + eps * var

    def encode(self, x, y, slice_idx):
        x = torch.flatten(x, start_dim=1)
        if x.device != self.embedding_device:
            x = x.to(self.embedding_device)
        x = torch.cat((x, y, slice_idx), dim=1)
        x = self.encoder(x.to(self.encoder.devices[0]))
    
        if x.device != self.reparameterization_device:
            x = x.to(self.reparameterization_device)

        return self.compute_mean(x), self.compute_log_var(x)

    def decode(self, z, y, slice_idx):
        if z.device != self.decoder.devices[0]:
            z = z.to(self.decoder.devices[0])
        z = torch.cat((z, y.to(self.decoder.devices[0]), slice_idx.to(self.decoder.devices[0])), dim=1)
        return self.decoder(z)

    def forward(self, x, y, slice_idx_normalized=None):
        y = self.label_embedding(y.to(self.embedding_device))
        slice_idx = torch.round(slice_idx_normalized * 9).long().to(self.embedding_device)
        slice_idx = self.slice_embedding(slice_idx)
        mean, log_var = self.encode(x, y, slice_idx)

        z = self.sample_from(mean, log_var)
        x_hat = self.decode(z, y, slice_idx)
        return x_hat.cuda(0), mean.cuda(0), log_var.cuda(0)

    def generate(self, y, slice_idx, batch_size: int=1):
        y = self.label_embedding(y)
        slice_idx = torch.tensor([slice_idx], device=self.embedding_device)
        slice_idx = self.slice_embedding(slice_idx)
        z = torch.randn((batch_size, self.num_dimensions), device=self.decoder.devices[0])
        return self.decode(z, y, slice_idx)
    
    def to_gpus(self):
        return self
    
def find_first_index(array, layer_idx):
    cumulative_sum = 0

    for idx, balance_value in enumerate(array):
        cumulative_sum += balance_value

        if cumulative_sum > layer_idx:
            return idx, layer_idx - cumulative_sum + balance_value

    # If the loop completes without finding a suitable index
    # This may happen if layer_idx is larger than the sum of all balance values
    raise ValueError("layer_idx is larger than the sum of all balance values")

    
class FairScaleVAEPlugIn(nn.Module):
    def __init__(self, vae, config: dict) -> None:
        super().__init__()
        print("splitting")
        print(vae)


        self.label_embedding = vae.label_embedding
        assert not hasattr(vae, 'slice_embedding')
    
        self.num_dimensions=vae.num_dimensions
        self.encoder = vae.encoder
        
        self.compute_mean = vae.compute_mean
        self.compute_log_var = vae.compute_log_var
        
        self.decoder = vae.decoder
        

        self.encoder = fairscale.nn.Pipe(self.encoder, checkpoint='never', devices=config['encoder devices'], 
                                         balance=config['encoder balance'])
        

        self.decoder = fairscale.nn.Pipe(self.decoder, checkpoint='never', devices=config['decoder devices'], 
                                         balance=config['decoder balance'])
        
        self.embedding_device = config['embedding device']
        self.reparameterization_device = config['reparameterization device']
        self.label_embedding.cuda(self.embedding_device)
        self.compute_log_var.cuda(self.reparameterization_device)
        self.compute_mean.cuda(self.reparameterization_device)
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        old_to_new_key = {}

        for key in state_dict:
            if key.startswith("encoder") or key.startswith("decoder"):
                encoder_decoder_label = key.split(".")[0]
                if(encoder_decoder_label == "encoder"):
                    layer = self.encoder
                else:
                    assert encoder_decoder_label == "decoder"
                    layer = self.decoder

                # layer = eval(f"self.{encoder_decoder_label}")
                layer_idx = int(key.split(".")[1])
                #find the first index in self.encoder.balance such that the sum of all previous elements is larger than layer_idx
                partition_idx, idx_within_partition = find_first_index(layer.balance, layer_idx)

                new_key = f"{encoder_decoder_label}.partitions.{partition_idx}.{layer_idx}." + ".".join(key.split(".")[2:])
                old_to_new_key[key] = new_key
                #encoder.0.weight -> encoder.partitions.0.0.weight
        

        for old_key, new_key in old_to_new_key.items():
            state_dict[new_key] = state_dict.pop(old_key)

        return super().load_state_dict(state_dict, strict)


    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        old_to_new_key = {}
        super_dict = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        for key in super_dict:
            if key.startswith("encoder") or key.startswith("decoder"):
                # encoder.partitions.0.0.weight -> encoder.0.weight
                encoder_decoder_label = key.split(".")[0]
                if(encoder_decoder_label == "encoder"):
                    layer = self.encoder
                else:
                    assert encoder_decoder_label == "decoder"
                    layer = self.decoder

                assert key.split(".")[1] == "partitions"
                partition_idx = int(key.split(".")[2])
                layer_idx = int(key.split(".")[3])



                new_key = f"{encoder_decoder_label}.{layer_idx}." + ".".join(key.split(".")[4:])
                old_to_new_key[key] = new_key
        
        for old_key, new_key in old_to_new_key.items():
            super_dict[new_key] = super_dict.pop(old_key)
        
        return super_dict




    def sample_from(self, mean, log_var):
        if not self.training:
            return mean
        eps = torch.randn(mean.shape, device=mean.device)
        var =  torch.exp(0.5 *log_var)
        return mean + eps * var

    def encode(self, x, y, slice_idx=None):
        x = torch.flatten(x, start_dim=1)
        if x.device != self.embedding_device:
            x = x.to(self.embedding_device)
        x = torch.cat((x, y), dim=1)
        x = self.encoder(x.to(self.encoder.devices[0]))
    
        if x.device != self.reparameterization_device:
            x = x.to(self.reparameterization_device)

        return self.compute_mean(x), self.compute_log_var(x)

    def decode(self, z, y, slice_idx=None):
        if z.device != self.decoder.devices[0]:
            z = z.to(self.decoder.devices[0])
        z = torch.cat((z, y.to(self.decoder.devices[0])), dim=1)
        return self.decoder(z)

    def forward(self, x, y, slice_idx_normalized=None):
        y = self.label_embedding(y.to(self.embedding_device))
        mean, log_var = self.encode(x, y)

        z = self.sample_from(mean, log_var)
        x_hat = self.decode(z, y)
        return x_hat.cuda(0), mean.cuda(0), log_var.cuda(0)

    def generate(self, y, slice_idx=None, batch_size: int=1):
        y = self.label_embedding(y)
        z = torch.randn((batch_size, self.num_dimensions), device=self.decoder.devices[0])
        return self.decode(z, y)
    
    def to_gpus(self):
        return self
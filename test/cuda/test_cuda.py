import torch

def test_cuda():
    nr_devices = torch.cuda.device_count()
    assert nr_devices > 0
    device = nr_devices-1  # Last device chosen
    torch.cuda.set_device(device)
    tensor = torch.zeros((2,3)).cuda()
    assert str(tensor.device) == 'cuda:'+str(device)
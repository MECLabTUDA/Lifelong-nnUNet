from nnunet_ext.network_architecture.MultiHead_Module import MultiHead_Module
from nnunet.network_architecture.generic_UNet import Generic_UNet
from batchgenerators.utilities.file_and_folder_operations import *
import torch

def main():
    # import os, shutil
    # import numpy as np
    # import SimpleITK as sitk
    # base = '/gris/gris-f/homestud/aranem/Lifelong-nnUNet-storage/nnUNet_raw/Task90_HipJoined/labelsTr'
    # base_new = '/gris/gris-f/homestud/aranem/Lifelong-nnUNet-storage/nnUNet_raw/Task90_HipJoined_2d/labelsTr'
    # for path in os.listdir(base):
    #     if '._' in path or 'DS_Store' in path:
    #         continue
    #     if 'hippocampus' in path:
    #         shutil.copy(os.path.join(base, path), os.path.join(base_new, path))
    #         continue
    #     img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(base, path)))
    #     print(path)
    #     print("Before: "+str(img.shape))
    #     img = np.transpose(img, (2, 0, 1))
    #     # img = torch.from_numpy(img).float().permute(2, 0, 1)
    #     # img = img.numpy()
    #     print("After: "+str(img.shape))
    #     sitk.WriteImage(sitk.GetImageFromArray(img), os.path.join(base_new, path))


    f = load_pickle('/gris/gris-f/homestud/aranem/Lifelong-nnUNet-storage/nnUNet_trained_models/nnUNet_ext/2d/Task099_HarP_Task098_Dryad_Task097_DecathHip/metadata/Generic_UNet/SEQ/rw_data/param_values.pkl')
    # print(f['Task09[8_Dryad'].values())
    for name, h_old in f['Task099_HarP'].items():
      # print(h_old)
      print(name)
      # raise

    # import copy
    # import numpy as np
    # a = np.array([[[1, 2, 3, 4],
    #          [5, 6, 7, 8],
    #          [1, 2, 3, 4],
    #          [8, 8, 7, 5]],
    #          [[1, 2, 3, 4],
    #          [1, 8, 3, 4],
    #          [1, 1, 9, 4],
    #          [1, 2, 0, 4]],
    #          [[1, 21, 3, 4],
    #          [1, 2, 3, 4],
    #          [1, 23, 3, 4],
    #          [1, 2, 3, 54]]])

    # a = a.reshape(4, 4, 3)
    # # H, W, _ = a.shape
    # _a = torch.FloatTensor(a)
    # _c = torch.mean(_a, 0)  # H × C width-pooled slices of _a
    # _d = torch.mean(_a, 1)  # W × C height-pooled slices of _a
    # # c = np.add.reduce(copy.deepcopy(a), 0)/H
    # # d = np.add.reduce(copy.deepcopy(a), 1)/W

    # # b = a[0, :, :]
    # # for i in range(1, 4):
        
    # #     b+=a[i, :, :]
        
    # # print(b/H)
    # print(_c)
    # print(_d)

    # print(torch.cat((_c, _d)).size())


             

    # split = '     conv_blocks_context    . 0 .    blocks .    1'
    # task = 'Task_A'
    # model = MultiHead_Module(Generic_UNet, split, task, prev_trainer=None, input_channels=3, base_num_features=5, num_classes=2, num_pool=3)
    # model.add_new_task('Task_B', False)
    # a = model.assemble_model('Task_B')
    # names = [name for name, module in a.named_modules() if 'conv.Conv' in str(type(module))]
    # for name, mo in a.named_modules():
    #   print(isinstance(mo, torch.nn.conv))
    
    # print(names)
    # print(a)

if __name__ == "__main__":
    main()

# Split still not that perfect for conv_blocks_context.1.1.blocks.0, ie. conv_blocks_context.1.1
""" GenericUNet using input_channels=3, base_num_features=5, num_classes=2, num_pool=3:
Generic_UNet(
  (conv_blocks_localization): ModuleList(
    (0): Sequential(
      (0): StackedConvLayers(
        (blocks): Sequential(
          (0): ConvDropoutNormNonlin(
            (conv): Conv2d(40, 20, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1])
            (instnorm): BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
          )
        )
      )
      (1): StackedConvLayers(
        (blocks): Sequential(
          (0): ConvDropoutNormNonlin(
            (conv): Conv2d(20, 10, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1])
            (instnorm): BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
          )
        )
      )
    )
    (1): Sequential(
      (0): StackedConvLayers(
        (blocks): Sequential(
          (0): ConvDropoutNormNonlin(
            (conv): Conv2d(20, 10, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1])
            (instnorm): BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
          )
        )
      )
      (1): StackedConvLayers(
        (blocks): Sequential(
          (0): ConvDropoutNormNonlin(
            (conv): Conv2d(10, 5, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1])
            (instnorm): BatchNorm2d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
          )
        )
      )
    )
    (2): Sequential(
      (0): StackedConvLayers(
        (blocks): Sequential(
          (0): ConvDropoutNormNonlin(
            (conv): Conv2d(10, 5, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1])
            (instnorm): BatchNorm2d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
          )
        )
      )
      (1): StackedConvLayers(
        (blocks): Sequential(
          (0): ConvDropoutNormNonlin(
            (conv): Conv2d(5, 5, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1])
            (instnorm): BatchNorm2d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
          )
        )
      )
    )
  )
  (conv_blocks_context): ModuleList(
    (0): StackedConvLayers(
      (blocks): Sequential(
        (0): ConvDropoutNormNonlin(
          (conv): Conv2d(3, 5, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1])
          (dropout): Dropout2d(p=0.5, inplace=True)
          (instnorm): BatchNorm2d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
        )
        (1): ConvDropoutNormNonlin(
          (conv): Conv2d(5, 5, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1])
          (dropout): Dropout2d(p=0.5, inplace=True)
          (instnorm): BatchNorm2d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
        )
      )
    )
    (1): StackedConvLayers(
      (blocks): Sequential(
        (0): ConvDropoutNormNonlin(
          (conv): Conv2d(5, 10, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1])
          (dropout): Dropout2d(p=0.5, inplace=True)
          (instnorm): BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
        )
        (1): ConvDropoutNormNonlin(
          (conv): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1])
          (dropout): Dropout2d(p=0.5, inplace=True)
          (instnorm): BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
        )
      )
    )
    (2): StackedConvLayers(
      (blocks): Sequential(
        (0): ConvDropoutNormNonlin(
          (conv): Conv2d(10, 20, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1])
          (dropout): Dropout2d(p=0.5, inplace=True)
          (instnorm): BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
        )
        (1): ConvDropoutNormNonlin(
          (conv): Conv2d(20, 20, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1])
          (dropout): Dropout2d(p=0.5, inplace=True)
          (instnorm): BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
        )
      )
    )
    (3): Sequential(
      (0): StackedConvLayers(
        (blocks): Sequential(
          (0): ConvDropoutNormNonlin(
            (conv): Conv2d(20, 40, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1])
            (dropout): Dropout2d(p=0.5, inplace=True)
            (instnorm): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
          )
        )
      )
      (1): StackedConvLayers(
        (blocks): Sequential(
          (0): ConvDropoutNormNonlin(
            (conv): Conv2d(40, 20, kernel_size=(3, 3), stride=(1, 1), padding=[1, 1])
            (dropout): Dropout2d(p=0.5, inplace=True)
            (instnorm): BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (lrelu): LeakyReLU(negative_slope=0.01, inplace=True)
          )
        )
      )
    )
  )
  (td): ModuleList(
    (0): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (1): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
    (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  )
  (tu): ModuleList(
    (0): Upsample()
    (1): Upsample()
    (2): Upsample()
  )
  (seg_outputs): ModuleList(
    (0): Conv2d(10, 2, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (1): Conv2d(5, 2, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (2): Conv2d(5, 2, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
)
"""



# from nnunet.network_architecture.generic_UNet import Generic_UNet
# from nnunet.network_architecture.initialization import InitWeights_He
# from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
# from torch import nn
# from torch.cuda.amp import autocast, GradScaler
# from torch.optim import SGD
# import numpy as np
# import torch


# def dummy_dl(batch_size, patch_size, data_channels, num_classes):
#     data = np.random.random((batch_size, data_channels, *patch_size))
#     seg = np.random.uniform(0, num_classes-1, (batch_size, 1, *patch_size)).round()
#     data_torch = torch.from_numpy(data).float()
#     seg_torch = torch.from_numpy(seg).float()
#     while True:
#         yield data_torch, seg_torch


# def main():
#     patch_size = [320, 256]
#     batch_size = 40
#     data_channels = 1
#     num_classes = 2

#     model = Generic_UNet(data_channels, 16, num_classes, 6, 2, 2, nn.Conv2d, nn.InstanceNorm2d,
#                          {'eps': 1e-05, 'affine': True}, nn.Dropout2d,
#                          {'p': 0, 'inplace': True}, nn.LeakyReLU, {'negative_slope': 0.01, 'inplace': True}, False,
#                          False, lambda x: x, InitWeights_He(1e-2), [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
#                          [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]], False, True, True, 512)
#     model = model.cuda()
#     optimizer = SGD(model.parameters(), 0.01, 0.99, nesterov=True, weight_decay=3e-5)

#     dl = dummy_dl(batch_size, patch_size, data_channels, num_classes)
#     dl_val = dummy_dl(batch_size, patch_size, data_channels, num_classes)

#     loss_fn = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})

#     scaler = GradScaler()
#     for ep in range(1000):
#         print(ep)
#         losses_train = []
#         for batch in range(30):
#             optimizer.zero_grad()

#             data, seg = next(dl)
#             data = data.cuda()
#             seg = seg.cuda()

#             with autocast():
#                 output = model(data)
#                 loss = loss_fn(output, seg)

#             scaler.scale(loss).backward()
#             #loss.backward()

#             scaler.step(optimizer)
#             #optimizer.step()

#             scaler.update()

#             losses_train.append(loss.item())
#         print(np.mean(losses_train))

#         losses_val = []
#         with torch.no_grad():
#             for batch in range(30):
#                 data, seg = next(dl_val)
#                 data = data.cuda()
#                 seg = seg.cuda()

#                 with autocast(True):
#                 # with autocast(False):
#                     output = model(data)
#                     loss = loss_fn(output, seg)
#                 losses_val.append(loss.item())
#         print(np.mean(losses_val))


# if __name__ == "__main__":
#     main()

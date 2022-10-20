from torch import nn
from voxelmorph.torch.layers import SpatialTransformer
from voxelmorph.torch.networks import VxmDense as VoxelMorph_
        
class VoxelMorph(VoxelMorph_):
    def __init__(self, inshape, nb_unet_features=None, nb_unet_levels=None, unet_feat_mult=1,
                 int_steps=7, int_downsize=2, bidir=False, use_probs=False):
        r"""Initialize.
        """
        super().__init__(inshape, nb_unet_features, nb_unet_levels, unet_feat_mult, int_steps, int_downsize,
                         bidir, use_probs)
        # -- Add this so we don't get an error during training with the nnUNet pipeline -- #
        self.do_ds = None
        self._2d = len(inshape) == 2
        
    def forward(self, source, target, registration=False):
        r"""Adapt the forward pass accordingly so we don't get any errors.
        """
        # -- Get the inputs right -- #
        if self._2d and len(source.size()) == 3 or not self._2d and len(source.size()) == 4:
            source = source.unsqueeze(1)
        elif not self._2d and len(source.size()) == 3:
            source = source.unsqueeze(0).unsqueeze(1)
            
        if self._2d and len(target.size()) == 3 or not self._2d and len(target.size()) == 4:
            target = target.unsqueeze(1)
        elif not self._2d and len(target.size()) == 3:
            target = target.unsqueeze(0).unsqueeze(1)
            
        ret = super().forward(source, target, registration)
        return ret

    def get_device(self):
        if next(self.parameters()).device.type == "cpu":
            return "cpu"
        else:
            return next(self.parameters()).device.index
          
class Transform(nn.Module):
    """
    Simple transform model to apply dense or affine transforms.
    """
    def __init__(self, inshape, rescale=None):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            rescale: Transform rescale factor. Default is None.
        """
        super().__init__()
        # configure inputs
        self.transform = SpatialTransformer(inshape, mode='nearest')
        self.rescale = rescale
        
    def forward(self, source, flow):
        r"""Adapt the forward pass accordingly so we don't get any errors.
        """
        # -- Get the inputs right -- #
        moved_img = self.transform(source, flow)
        return moved_img
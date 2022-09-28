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
        
    def forward(self, source, target, registration=False):
        r"""Adapt the forward pass accordingly so we don't get any errors.
        """
        if len(source.size()) == 3:
            source = source.unsqueeze(1)
        if len(target.size()) == 3:
            target = target.unsqueeze(1)
        
        # -- Do changes here to get VoxelMorph running in nnUNet as well -- #
        ret = super().forward(source, target, registration)
        return ret
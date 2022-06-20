from torch import nn
from timm.models.layers.patch_embed import PatchEmbed as PatchEmbed2D

class PatchEmbed(PatchEmbed2D):
    r"""This class represents the three and two dimensional Patch Embedding based on the
        two dimensional one from the timm module.
    """
    def __init__(self, img_size=224, patch_size=16, img_depth=None, in_chans=3, embed_dim=768, norm_layer=None, flatten=True,
                embed2D=True, task_specific_ln=False, task_name=None):
        r"""Constructor of the class. Note that embed2D is already pre-set, otherwise the implementation of timm
            module would fail because an argument would be missing. This means when a three dimensional embedding
            is desired, the flag has to be set, otherwise there will be a mixup in the model building process
            which might -- in the best case -- result in an error!
        """
        # -- Create a two dimensional patch embedding -- #
        super(PatchEmbed, self).__init__(img_size, patch_size, in_chans, embed_dim, norm_layer, flatten)
        
        # -- Set flag to use 2D or 3D and if task specific LNs -- #
        self.embed2D = embed2D
        self.task_specific_ln = task_specific_ln

        # -- If the user wants it three dimensional, make corresponding adjustments -- #
        if not self.embed2D:
            # -- Check that the image depth is not None -- #
            assert img_depth is not None, 'Please provide the depth of the image when using a three dimensional patch embedding..'
            # -- Perform assertion checks wrt img_size and patch_size -- #
            assert len(img_size) == 2 and len(patch_size) == 2,\
            'Please provide the img_size and patch_size as two dimensional tuples (height, width)..'
            # -- Reset image size, patch size and num_patches since they now should be in a tuple of length 3 -- #
            # -- dimensions: (img_depth, height, width)
            self.img_size = (img_depth, *img_size)
            self.patch_size = (img_depth, *patch_size)
            # -- Include the depth into nr of patches so everything maps during the forward pass -- #
            self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (img_depth // patch_size[0])

            # -- Copied and modified from timm module -- #
            # -- Input for Conv3D has to be 5D: [batch_size, channels, img_depth, height, width] -- #
            self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])

        # -- If task specific norms, than do this for the patch embeddings as well -- #
        if task_specific_ln:
            self.norm = nn.ModuleDict()
            self.norm[task_name] = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x, task_name):
        r"""Represents the forward mechanism when called PatchEmbed(...)(x)."""
        # -- Copied from timm module -- #
        if self.embed2D:
            _, _, H, W = x.shape
            assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
            assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        else:
            # -- Now the shape needs to be five dimensional: [batch_size, channels, img_depth, height, width] -- #
            _, _, _, H, W = x.shape
            assert H == self.img_size[1], f"Input image height ({H}) doesn't match model ({self.img_size[1]})."
            assert W == self.img_size[2], f"Input image width ({W}) doesn't match model ({self.img_size[2]})."
        
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        if self.task_specific_ln:
            x = self.norm[task_name](x)
        else:
            x = self.norm(x)
        # -- Copied from timm module -- #

        return x
from torch import nn, einsum
from einops import rearrange
from nnunet_ext.utilities.helpful_functions import *
from timm.models.vision_transformer import Attention as AttentionTimm

class VanillaAttention(AttentionTimm):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., is_LSA=False, num_patches=16):
        # -- Do not modify the attention module if not LSA -- #
        qkv_bias = False if is_LSA else qkv_bias    # --> Overwrite this; in LSA bias is false, see https://github.com/aanna0701/SPT_LSA_ViT/blob/main/models/vit.py#L61
        super().__init__(dim, num_heads, qkv_bias, attn_drop, proj_drop)

        # -- Set LSA flag and make changes is LSA is true -- #
        self.LSA = is_LSA

        # -- Do the modifications so nothing will crash in forward function --> copied and modified from https://github.com/aanna0701/SPT_LSA_ViT/blob/main/models/vit.py#L51 -- #
        if self.LSA:
            head_dim = dim // num_heads
            inner_dim = head_dim *  num_heads   # --> not always dim, since we floor the result if its not even!
            project_out = not (num_heads == 1 and head_dim == dim)
            self.num_patches = num_patches
            self.heads = num_heads
            self.dim = dim
            self.inner_dim = inner_dim
            self.attend = nn.Softmax(dim = -1)
            self._init_weights(self.qkv)
            self.to_out = nn.Sequential(
                nn.Linear(self.inner_dim, self.dim),
                nn.Dropout(attn_drop)
            ) if project_out else nn.Identity()

            self.scale = nn.Parameter(self.scale*torch.ones(num_heads))    
            self.mask = torch.eye(self.num_patches+1, self.num_patches+1)
            self.mask = torch.nonzero((self.mask == 1), as_tuple=False)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x, cross_attn=False, use_q=None):
        if self.LSA:
            # -- Perform forward function from Attention with LSA --> copied and modified from https://github.com/aanna0701/SPT_LSA_ViT/blob/main/models/vit.py#L75 -- #
            b, _, _, h = *x.shape, self.heads
            qkv = self.qkv(x).chunk(3, dim = -1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
            q_ = q.clone()
            if use_q is not None:
                assert q.shape == use_q.shape, "The shape of the to be used querie should be equal to the one that should be replaced.."
                q = use_q
            
            scale = self.scale
            dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k), scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((b, h, 1, 1)))
            dots[:, :, self.mask[:, 0], self.mask[:, 1]] = -987654321

            attn = self.attend(dots)
            weights = attn
            out = einsum('b h i j, b h j d -> b h i d', attn, v) 
                
            out = rearrange(out, 'b h n d -> b n (h d)')
            x = self.to_out(out)
        else:
            # -- Do not modify the attention module if not LSA, only keep track of weights -- #
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
            q_ = q.clone()
            if use_q is not None:
                assert q.shape == use_q.shape, "The shape of the to be used querie should be equal to the one that should be replaced.."
                q = use_q
            
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            weights = attn
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        
        # -- Always return a copy of q as well for cross-attention -- #
        if cross_attn and use_q is None:
            return x, weights, q_
        else:
            return x, weights

class ScaleAttention(VanillaAttention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., is_LSA=False, num_patches=16):
        # -- Initialize same as any Attention -- #
        super().__init__(dim, num_heads, qkv_bias, attn_drop, proj_drop, is_LSA, num_patches)

        # -- Just add the lamb parameter -- #
        self.lamb = nn.Parameter(torch.zeros(num_heads), requires_grad=True)

    def forward(self, x, cross_attn=False, use_q=None):
        if self.LSA:    # <-- Test this, i.e. set LSA and AttnScale!!!
            B, N, C = x.shape
            # -- Perform forward function from Attention with LSA --> copied and modified from https://github.com/aanna0701/SPT_LSA_ViT/blob/main/models/vit.py#L75 -- #
            b, _, _, h = *x.shape, self.heads
            qkv = self.qkv(x).chunk(3, dim = -1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
            q_ = q.clone()
            if use_q is not None:
                assert q.shape == use_q.shape, "The shape of the to be used querie should be equal to the one that should be replaced.."
                q = use_q
            
            scale = self.scale
            dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k), scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((b, h, 1, 1)))
            dots[:, :, self.mask[:, 0], self.mask[:, 1]] = -987654321

            attn = self.attend(dots)
            weights = attn

            # -- Relevant part for AttentionScale method -- #
            attn_d = torch.ones(attn.shape[-2:], device=attn.device) / N    # [l, l]
            attn_d = attn_d[None, None, ...]                                # [B, N, l, l]
            attn_h = attn - attn_d                                          # [B, N, l, l]
            attn_h = attn_h * (1. + self.lamb[None, :, None, None])         # [B, N, l, l]
            attn = attn_d + attn_h                                          # [B, N, l, l]
            attn = self.attn_drop(attn)

            out = einsum('b h i j, b h j d -> b h i d', attn, v)     
            out = rearrange(out, 'b h n d -> b n (h d)')
            x = self.to_out(out)
        else:
            # -- Vanilla Attention with AttentionScale method -- #
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
            q_ = q.clone()
            if use_q is not None:
                assert q.shape == use_q.shape, "The shape of the to be used querie should be equal to the one that should be replaced.."
                q = use_q
            
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            weights = attn
            attn = self.attn_drop(attn)

            # -- Relevant part for AttentionScale method -- #
            attn_d = torch.ones(attn.shape[-2:], device=attn.device) / N    # [l, l]
            attn_d = attn_d[None, None, ...]                                # [B, N, l, l]
            attn_h = attn - attn_d                                          # [B, N, l, l]
            attn_h = attn_h * (1. + self.lamb[None, :, None, None])         # [B, N, l, l]
            attn = attn_d + attn_h                                          # [B, N, l, l]
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        
        # -- Always return a copy of q as well for cross-attention -- #
        if cross_attn and use_q is None:
            return x, weights, q_
        else:
            return x, weights
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from mmdet.models.builder import BACKBONES
from mmdet.utils import get_root_logger
from mmcv.runner import *
from timm.models.vision_transformer import Block as TimmBlock
from mmcv.runner import (auto_fp16, force_fp32,)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SeparableAttention(nn.Module):
    """
    Depthwise separable self-attention, including DSA and PSA.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ws=7):
        super(SeparableAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.ws = ws
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.DSA_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.DSA_attn_drop = nn.Dropout(attn_drop)

        self.win_tokens_norm = nn.LayerNorm(dim)
        self.win_tokens_act = nn.GELU()

        self.PSA_qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.PSA_attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    @auto_fp16()
    def forward(self, x, H, W):
        B, win_num, _, C = x.shape # B, win_num, win_size+1, C
        h_group, w_group = math.ceil(H / self.ws), math.ceil(W / self.ws) # padding
        win_size = self.ws * self.ws
        
        # Depthwise Self-Attention (DSA)
        DSA_qkv = self.DSA_qkv(x).reshape(B, win_num, -1, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5) # B, win_num, win_size+1, 3, n_head, head_dim -> 3, B, win_num, n_head, win_size+1, head_dim
        DSA_q, DSA_k, DSA_v = DSA_qkv[0], DSA_qkv[1], DSA_qkv[2]  # B, win_num, n_head, win_size+1, head_dim

        DSA_attn = (DSA_q @ DSA_k.transpose(-2, -1)) * self.scale  # Q@K = B, win_num, n_head, win_size+1, win_size+1
        DSA_attn = DSA_attn.softmax(dim=-1)
        DSA_attn = self.DSA_attn_drop(DSA_attn)
        attn_out = (DSA_attn @ DSA_v).transpose(2, 3).reshape(B, win_num, -1, C) # attn @ V --> B, win_num, n_head, win_size+1, C//n_head -> (t(2,3)) -> B, win_num, win_size+1, C
        attn_out = attn_out + x    # short cut

        # Pointwise Self-Attention (PSA)
        # slice window tokens (win_tokens) and feature maps (attn_x)
        attn_win_tokens = attn_out[:, :, 0, :] # B, win_num, C
        attn_x = attn_out[:, :, 1:, :] # B, win_num, win_size, C

        # LN & Act
        attn_win_tokens = self.win_tokens_norm(attn_win_tokens)
        attn_win_tokens = self.win_tokens_act(attn_win_tokens)

        PSA_qk = self.PSA_qk(attn_win_tokens).reshape(B, win_num, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4) # B, win_num, 2, n_head, head_dim -> 2, B, n_head, win_num, head_dim 
        PSA_q, PSA_k = PSA_qk[0], PSA_qk[1] # B, n_head, win_num, head_dim

        # resahpe attn_x to multi_head
        PSA_v = attn_x.reshape(B, win_num, win_size, self.num_heads, -1).permute(0, 3, 1, 2, 4) # B, win_num, win_size, n_head, head_dim -> B, n_head, win_num, win_size, head_dim
        PSA_v = PSA_v.reshape(B, self.num_heads, win_num, -1) # (B, n_head, win_num, win_size*head_dim)
        
        PSA_attn = (PSA_q @ PSA_k.transpose(-2, -1)) * self.scale # Q@K = B, n_head, win_num, win_num
        PSA_attn = PSA_attn.softmax(dim=-1)
        PSA_attn = self.PSA_attn_drop(PSA_attn)
        attn_out = (PSA_attn @ PSA_v) # (B, n_head, win_num, win_num) @ (B, n_head, win_num, win_size*head_dim) = (B, n_head, win_num, win_size*head_dim)
        
        # delete padding and reshape to B, N, C
        attn_out = attn_out.transpose(1, 2).reshape(B, win_num, self.num_heads, win_size, -1) # B, win_num, n_head, win_size*head_dim -> B, win_num, n_head, win_size, head_dim
        attn_out = attn_out.transpose(2,3).reshape(B, win_num, win_size, C) # B, win_num, win_size, C
        attn_out = attn_out + attn_x    # short cut
        attn_out = attn_out.reshape(B, h_group, w_group, self.ws, self.ws, C).transpose(2, 3).reshape(B, h_group * self.ws, w_group * self.ws, C) # B Hp Wp C
        
        if (h_group > H // self.ws) or (w_group >  W // self.ws): # delete padding
            attn_out = attn_out[:, :H, :W, :].contiguous() # B H W C
        attn_out = attn_out.reshape(B, -1, C) # B N C
    
        x = self.proj(attn_out)
        x = self.proj_drop(x) # B, N, C

        return x      


class SepViTBlock(TimmBlock):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, ws=7, stage=0):
        super(SepViTBlock, self).__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                                         drop_path, act_layer, norm_layer)
        del self.attn
        self.ws = ws
        self.attn = SeparableAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, ws)

    def forward(self, x, H, W):
        B, N, C = x.shape
        # padding
        attn_x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        if pad_r > 0 or pad_b > 0:
            attn_x = F.pad(attn_x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            
        h_group, w_group = math.ceil(H / self.ws), math.ceil(W / self.ws)
        assert attn_x.shape[1] == h_group * self.ws, 'The wrong padding.'

        attn_x = attn_x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3).reshape(B, h_group * w_group, self.ws * self.ws, C) # B, win_num, win_size, C
        win_tokens = torch.zeros((B, h_group * w_group, 1, C), device=x.device) # fixed zero initialization
        attn_x = torch.cat((win_tokens, attn_x), dim=2) # B, win_num, win_size+1, C

        x = x + self.drop_path(self.attn(self.norm1(attn_x), H, W)) # B, N, C
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
    
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        if patch_size[0] == 4 or patch_size[0] == 2: # non-overlapping, no padding
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        else: # overlapping patch embedding, padding
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=(patch_size[0] // 2, patch_size[1] // 2)) 
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, (H, W)


class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.s = s
    
    @auto_fp16()
    def forward(self, x, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]


class SepViT(nn.Module):
    def __init__(self, img_size=224, patch_size=7, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256],
                 num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[4, 4, 4], block_cls=SepViTBlock, ws=[7], extra_norm=False):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        self.extra_norm = extra_norm
        if self.extra_norm:
            self.norm_list = nn.ModuleList()
            for dim in embed_dims:
                self.norm_list.append(norm_layer(dim))

        # patch_embed
        self.patch_embeds = nn.ModuleList()
        self.pos_drops = nn.ModuleList()
        for i in range(len(depths)):
            if i == 0:
                self.patch_embeds.append(PatchEmbed(img_size=img_size, patch_size=patch_size, stride=4, in_chans=in_chans, embed_dim=embed_dims[i]))
            else: 
                self.patch_embeds.append(PatchEmbed(img_size=img_size // 4 // 2 ** (i - 1), patch_size=3, stride=2, in_chans=embed_dims[i - 1], embed_dim=embed_dims[i]))
            self.pos_drops.append(nn.Dropout(p=drop_rate))

        # PEG
        self.pos_block = nn.ModuleList([PosCNN(embed_dim, embed_dim) for embed_dim in embed_dims])

        # SepViT Block
        cur = 0
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.blocks = nn.ModuleList()
        for k in range(len(depths)):
            _block = nn.ModuleList([block_cls(
                    dim=embed_dims[k], num_heads=num_heads[k], mlp_ratio=mlp_ratios[k], qkv_bias=qkv_bias,
                    qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], 
                    norm_layer=norm_layer, ws=ws[k][i], stage=k) for i in range(depths[k])])
            self.blocks.append(_block)
            cur += depths[k]

        # classifier
        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
    
        self.apply(self._init_weights)

    
    def _init_weights(self, m):
        import math
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            self.apply(self._init_weights)
            logger = get_root_logger()
            checkpoint = _load_checkpoint(pretrained, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            # load state_dict
            load_state_dict(self, state_dict, strict=False, logger=logger)
            # load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
        elif pretrained is None:
            self.apply(self._init_weights)
        else:
            raise TypeError('pretrained must be a str or None')
    
    @auto_fp16()
    def forward(self, x):
        B = x.shape[0]
        outputs = list()

        for i in range(len(self.depths)):
            x, (H, W) = self.patch_embeds[i](x)
            x = self.pos_drops[i](x)
            for j, blk in enumerate(self.blocks[i]):
                x = blk(x, H, W)
                if j == 0:
                    x = self.pos_block[i](x, H, W)  # PEG
            if self.extra_norm:
                x = self.norm_list[i](x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outputs.append(x)

        return outputs

           
def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict



@BACKBONES.register_module()
class SepViT_Tiny(SepViT):
    def __init__(self, **kwargs):
        super(SepViT_Tiny, self).__init__(
            patch_size=7, embed_dims=[96, 192, 384, 768], num_heads=[3, 6, 12, 24], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[1, 2, 6, 2], ws=[[7, 7], [7, 14], [7, 14]*3, [7, 7]],
            extra_norm=True, drop_path_rate=0.2
        )


@BACKBONES.register_module()
class SepViT_Small(SepViT):
    def __init__(self, **kwargs):
        super(SepViT_Small, self).__init__(
            patch_size=7, embed_dims=[96, 192, 384, 768], num_heads=[3, 6, 12, 24], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[1, 2, 14, 2], ws=[[7, 7], [7, 14], [7, 14]*7, [7, 7]],
            extra_norm=True, drop_path_rate=0.3
        )


@BACKBONES.register_module()
class SepViT_Base(SepViT):
    def __init__(self, **kwargs):
        super(SepViT_Base, self).__init__(
            patch_size=7, embed_dims=[128, 256, 512, 1024], num_heads=[4, 8, 16, 32], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[1, 2, 14, 2], ws=[[7, 7], [7, 14], [7, 14]*7, [7, 7]],
            extra_norm=True, drop_path_rate=0.5
        )










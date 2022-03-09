# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


""" Define the attention layers. """
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from scipy.optimize import linear_sum_assignment
import math


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        # self.class_token_pos = nn.Parameter(torch.zeros(1, 1, num_pos_feats * 2))
        # self.class_token_pos

    def forward(self, x, class_token=False):
        # x: b,d,h,w
        num_feats = x.shape[1]
        num_pos_feats = num_feats // 2
        # mask = tensor_list.mask
        mask = torch.zeros(x.shape[0], x.shape[2], x.shape[3], device=x.device).to(torch.bool)
        batch = mask.shape[0]
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-5
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2).contiguous()
        '''
        pos_x: b ,h, w, d//2
        pos_y: b, h, w, d//2
        pos: b, d, h, w
        '''
        if class_token:
            # pos = torch.cat((self.class_token_pos.repeat(batch, 1, 1), pos), dim=1)
            # pos = torch.cat((torch.mean(pos, dim=1, keepdim=True), pos), dim=1)
            pos = torch.cat((torch.zeros(batch, 1, pos.shape[2], dtype=pos.dtype, device=pos.device), pos), dim=1)
        return pos

class PositionEmbeddingSine2(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        # self.class_token_pos = nn.Parameter(torch.zeros(1, 1, num_pos_feats * 2))
        # self.class_token_pos

    def forward(self, x):
        # x: b, h, w, d
        num_feats = x.shape[3]
        num_pos_feats = num_feats // 2
        # mask = tensor_list.mask
        mask = torch.zeros(x.shape[0], x.shape[1], x.shape[2], device=x.device).to(torch.bool)
        batch = mask.shape[0]
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-5
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # pos = torch.cat((pos_y, pos_x), dim=3).flatten(1, 2)
        pos = torch.cat((pos_y, pos_x), dim=3).contiguous()
        '''
        pos_x: b ,h, w, d//2
        pos_y: b, h, w, d//2
        pos: b, h, w, d
        '''
        return pos

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (tuple): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (tuple): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


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


class SelfIIWA(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0.1, proj_drop=0.1, mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 drop_path=0.1):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # self.resolution = resolution

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        #self.relative_position_bias_table2 = nn.Parameter(
        #    torch.zeros((2 * 50 - 1) * (2 * 50 - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # get pair-wise relative position index for each window
        # self.max_windows = 20
        # coords_h = torch.arange(self.max_windows)
        # coords_w = torch.arange(self.max_windows)
        # coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        # coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        # relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        # relative_coords[:, :, 0] += self.max_windows - 1  # shift to start from 0
        # relative_coords[:, :, 1] += self.max_windows - 1
        # relative_coords[:, :, 0] *= 2 * self.max_windows - 1
        # relative_position_index_window = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # self.register_buffer("relative_position_index_window", relative_position_index_window)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv2 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Sequential(
        #    nn.Linear(2 * dim, dim),
        # nn.LayerNorm(dim),
        # nn.GELU()
        # )
        self.proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )
        self.proj2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_drop2 = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        # self.avgpool = nn.AdaptiveAvgPool2d((self.resolution[0]//self.window_size[0],
        #                                      self.resolution[1]//self.window_size[1]))
        # self.avgpool = nn.Sequential(
        #     nn.AvgPool2d(kernel_size=(self.window_size[0], self.window_size[1])),
        #     nn.Conv2d(dim, dim, 1)
        # )
        self.avgpool = nn.AvgPool2d(kernel_size=(self.window_size[0], self.window_size[1]))

        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=proj_drop)
        self.pose_encoding = PositionEmbeddingSine(normalize=True)
        # self.f_lepe = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (B, H, W, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # pad feature maps to multiples of window size
        B, H, W, C = x.shape
        # shortcut = x
        pad_l = pad_t = 0
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        shortcut = x

        x = window_partition(x, self.window_size)  # # nW*B, window_size, window_size, C
        C = x.shape[-1]
        x = x.view(-1, self.window_size[0] * self.window_size[1], C)  # nW*B, window_size*window_size, C
        # H, W = self.resolution
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # lepe = self.get_lepe(v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        # window attention
        x = x.view(-1, self.window_size[0], self.window_size[1], C)
        # B = int(B_ / (Hp * Wp / self.window_size[0] / self.window_size[1]))
        x = x.view(B, Hp // self.window_size[0], Wp // self.window_size[1], self.window_size[0], self.window_size[1],
                   -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
        x = shortcut + x
        x = x + self.proj_drop(self.proj(x))
        shortcut = x

        xg = x.permute(0, 3, 1, 2).contiguous()  # B, C, Hp, Wp
        xg = self.avgpool(xg)
        w_size = (xg.shape[2] ** 2, xg.shape[3] ** 2)
        xg = xg + self.pose_encoding(xg)
        xg = xg.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        num_w = xg.shape[1]
        qkv = self.qkv2(xg).reshape(B, num_w, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                              4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        # relative_position_bias = self.relative_position_bias_table2[self.relative_position_index_window.view(-1)].view(
        #     self.max_windows * self.max_windows, self.max_windows * self.max_windows, -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0) # nH, Wh*Ww, Wh*Ww
        # relative_position_bias = F.adaptive_avg_pool2d(relative_position_bias, output_size=w_size)
        # attn = attn + relative_position_bias
        attn = self.softmax(attn)
        xg = (attn @ v).transpose(1, 2).reshape(B, num_w, C)
        xg = xg.transpose(1, 2).reshape(B, C, Hp // self.window_size[0], Wp // self.window_size[1])
        xg = F.interpolate(xg, size=(Hp, Wp), mode='bilinear', align_corners=True)
        xg = xg.permute(0, 2, 3, 1).contiguous()  # (B, Hp, Wp, C)
        # x = x + xg.permute(0, 2, 3, 1).contiguous() # (B, Hp, Wp, C)
        x = shortcut + xg
        x = x + self.proj_drop2(self.proj2(x))

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
            # xg = xg[:, :H, :W, :].contiguous()
        # x = torch.cat([x, xg], dim=-1)
        # x = x.view(B, -1, C)
        # x = self.proj(x)
        # x = self.proj_drop(x)

        # FFN
        # x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm1(x)))
        x = self.norm2(x)  # B, H, W, C
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class MS_CrossIIWA(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size_q, window_size_k, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0.1, proj_drop=0., mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 drop_path=0.1):

        super().__init__()
        self.dim = dim
        self.window_size_q = window_size_q  # Wh, Ww
        self.window_size_k = window_size_k  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # self.resolution = resolution

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size_q[0] - 1) * (2 * window_size_k[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size_q[0])
        coords_w = torch.arange(self.window_size_k[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size_q[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size_k[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size_q[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.qkv2 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_lin = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_lin = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_lin = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_lin2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_lin2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_lin2 = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Sequential(
        #     nn.Linear(dim, dim),
        #     nn.LayerNorm(dim),
        #     nn.GELU()
        # )
        self.proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )
        self.proj2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_drop2 = nn.Dropout(proj_drop)
        # self.proj_drop = nn.Dropout(proj_drop)

        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        # self.avgpool = nn.AdaptiveAvgPool2d((self.resolution[0]//self.window_size[0],
        #                                      self.resolution[1]//self.window_size[1]))
        self.pos_enc = PositionEmbeddingSine2(normalize=True)
        self.avgpool_q = nn.AvgPool2d(kernel_size=(self.window_size_q[0], self.window_size_q[1]))
        self.avgpool_k = nn.AvgPool2d(kernel_size=(self.window_size_k[0], self.window_size_k[1]))

        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=proj_drop)

    def forward(self, q, k, kg, mask=None):
        """
        Args:
            q: query features with shape of (B, H, W, C)
            k: key features List [ (B, H, W, C); (B, H/2, W/2, C);  (B, H/4, W/4, C)]
        """
        # pad feature maps to multiples of window size
        B, H1, W1, C = q.shape
        # B, H2, W2, C = k.shape

        pad_l = pad_t = 0
        pad_r = (self.window_size_q[1] - W1 % self.window_size_q[1]) % self.window_size_q[1]
        pad_b = (self.window_size_q[0] - H1 % self.window_size_q[0]) % self.window_size_q[0]
        q = F.pad(q, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, qHp, qWp, _ = q.shape
        shortcut = q
        qg = self.avgpool_q(q.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous().view(B, -1,
                                                                                                      C)  # B, num_window_q, C
        num_window_q = qg.shape[1]
        q = window_partition(q, self.window_size_q)  # # nWq*B, window_size_q, window_size_q, C
        # kg = []
        # for i in range(len(k)):
        #     pad_l = pad_t = 0
        #     h_tmp, w_tmp = k[i].shape[1: 3]
        #     pad_r = (self.window_size_k[1] - w_tmp % self.window_size_k[1]) % self.window_size_k[1]
        #     pad_b = (self.window_size_k[0] - h_tmp % self.window_size_k[0]) % self.window_size_k[0]
        #     k_tmp = F.pad(k[i], (0, 0, pad_l, pad_r, pad_t, pad_b))
        #     kg.append(self.avgpool_k(k_tmp.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous().view(B, -1, C))
        #     k[i] = window_partition(k_tmp, self.window_size_k) \
        #         .view(B, -1, self.window_size_k[0], self.window_size_k[1], C)  # B, nWk, window_size_k, window_size_k, C
        #     # _, kHp, kWp, _ = k.shape
        kg = torch.cat(kg, dim=1)  # B, num_window_k, C
        num_window_k = kg.shape[1]
        k = torch.cat(k, dim=1)  # B, num_window_k, window_size_k, window_size_k, C

        # window match
        sim_mat = qg @ kg.transpose(-2, -1)  # B, num_window_q, num_window_k
        indices = []
        for i in range(sim_mat.shape[0]):
            indices.append(linear_sum_assignment(sim_mat[i].detach().cpu(), True))
        indices = [torch.as_tensor(j, dtype=torch.int64, device=k.device) for i, j in indices]
        k = torch.stack([k[i].index_select(dim=0, index=indices[i]) for i in range(len(indices))], dim=0).contiguous()
        # B, num_window_q, window_size_k, window_size_k, C
        kg = torch.stack([kg[i].index_select(dim=0, index=indices[i]) for i in range(len(indices))], dim=0).contiguous()
        # B, num_window_q, C
        Hg, Wg = qHp // self.window_size_q[0], qWp // self.window_size_q[1]
        kg = kg.reshape(B, Hg, Wg, C)
        kg = kg + self.pos_enc(kg)
        kg = kg.view(B, -1, C).contiguous()

        k = k.view(-1, self.window_size_k[0], self.window_size_k[1],
                   C)  # # B*num_window_q, window_size_k, window_size_k, C
        # for i in range(len(k)):
        #     k[i] = window_partition(k[i], self.window_size_k)
        # C = x.shape[-1]
        q = q.view(-1, self.window_size_q[0] * self.window_size_q[1], C)  # nWq*B, window_size*window_size, C
        k = k.view(-1, self.window_size_k[0] * self.window_size_k[1], C)  # nWq*B, window_size*window_size, C
        # H, W = self.resolution
        B_, N1, C = q.shape
        B_, N2, C = k.shape
        q = self.q_lin(q).reshape(B_, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k2 = self.k_lin(k).reshape(B_, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        v = self.v_lin(k).reshape(B_, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k = k2
        # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size_q[0] * self.window_size_q[1], self.window_size_k[0] * self.window_size_k[1],
            -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N1, N2) + mask.unsqueeze(1).unsqueeze(0)

            attn = attn.view(-1, self.num_heads, N1, N2)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        q = (attn @ v).transpose(1, 2).reshape(B_, N1, C)

        # window attention
        q = q.view(-1, self.window_size_q[0], self.window_size_q[1], C)
        # B = int(B_ / (Hp * Wp / self.window_size[0] / self.window_size[1]))
        q = q.view(B, qHp // self.window_size_q[0], qWp // self.window_size_q[1], self.window_size_q[0],
                   self.window_size_q[1], -1)
        q = q.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, qHp, qWp, -1)
        #print(q.shape)
        # exit()
        q = shortcut + q
        q = q + self.proj_drop(self.proj(q))
        shortcut = q
        # xg = x.permute(0, 3, 1, 2).contiguous()
        # xg = self.avgpool(xg).permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        # num_w = xg.shape[1]
        # qkv = self.qkv2(xg).reshape(B, num_w, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        # q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        qg = self.avgpool_q(q.permute(0, 3, 1, 2).contiguous())
        qg = qg.permute(0, 2, 3, 1).contiguous()
        qg = qg + self.pos_enc(qg)
        qg = qg.view(B, -1, C)
        qg = self.q_lin2(qg).reshape(B, num_window_q, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                                   3).contiguous()
        kg2 = self.k_lin2(kg).reshape(B, num_window_q, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                                    3).contiguous()
        vg = self.v_lin2(kg).reshape(B, num_window_q, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                                   3).contiguous()
        kg = kg2

        qg = qg * self.scale
        attn = (qg @ kg.transpose(-2, -1))
        attn = self.softmax(attn)
        qg = (attn @ vg).transpose(1, 2).reshape(B, num_window_q, C)
        qg = qg.transpose(1, 2).reshape(B, C, qHp // self.window_size_q[0], qWp // self.window_size_q[1])
        qg = F.interpolate(qg, size=(qHp, qWp), mode='bilinear', align_corners=True)
        # q = q + qg.permute(0, 2, 3, 1).contiguous() # (B, qHp, qWp, C)
        qg = qg.permute(0, 2, 3, 1).contiguous()  # (B, qHp, qWp, C)
        q = shortcut + qg
        q = q + self.proj_drop2(self.proj2(q))

        if pad_r > 0 or pad_b > 0:
            q = q[:, :H1, :W1, :].contiguous()

        # FFN
        # q = shortcut + self.drop_path(q)
        q = q + self.drop_path(self.mlp(self.norm1(q)))
        q = self.norm2(q)  # B, H, W, C
        return q

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size_q={self.window_size_q}, ' \
               f'window_size_k={self.window_size_k}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


if __name__ == '__main__':
    # arr = np.random.random((3,3))
    arr = torch.rand(5, 4, 6)
    indices = []
    for i in range(arr.shape[0]):
        indices.append(linear_sum_assignment(arr[i], True))
    indices = [torch.as_tensor(j, dtype=torch.int64) for i, j in indices]

    wa = WindowAttention(256, window_size=(7, 7), num_heads=8, )
    wa = WindowCrossAttention(
        dim=64,
        window_size_q=(7, 7),
        window_size_k=(7, 7),
        num_heads=8,
    )
    x = torch.rand(1, 56, 56, 64)
    y = [torch.rand(1, 56, 56, 64), torch.rand(1, 28, 28, 64), torch.rand(1, 14, 14, 64)]
    z = wa(x, y)
    pass

import warnings
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from mmcv.cnn import ConvModule, xavier_init
# from mmcv.runner import auto_fp16

from ..builder import NECKS

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

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.,
                 activation="gelu"):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                     memory_mask=None,
                     memory_key_padding_mask=None,
                     pos=None,
                     query_pos=None):
        query = self.with_pos_embed(tgt, query_pos) # h*w, b, d
        key = self.with_pos_embed(memory, pos) # h*w, b, d

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class CrossAttentionLayer2(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.,
                 activation="gelu", normalize_before=False, kdim=256, vdim=256):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, kdim=kdim, vdim=vdim)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask=None,
                     memory_mask=None,
                     tgt_key_padding_mask=None,
                     memory_key_padding_mask=None,
                     pos=None,
                     query_pos=None):
        # q = k = self.with_pos_embed(tgt, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0]
        # tgt = tgt + self.dropout1(tgt2)
        # tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask=None,
                    memory_mask=None,
                    tgt_key_padding_mask=None,
                    memory_key_padding_mask=None,
                    pos=None,
                    query_pos=None):
        tgt2 = self.norm2(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0]
        # tgt = tgt + self.dropout1(tgt2)
        # tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                pos=None,
                query_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransFusion(nn.Module):  # add
    def __init__(self, in_chans1, in_chans2, d_model=128, nhead=8, win_size_k=(12, 12), mlp_ratio=2,
                 win_size_q=(12, 12), dropout=0.1):
        super(TransFusion, self).__init__()

        self.win_size_q = win_size_q
        self.win_size_k = win_size_k
        self.nhead = nhead
        self.d_model = d_model
        head_dim = d_model // nhead
        self.scale = head_dim ** -0.5

        self.avgpool_q = nn.AvgPool2d(kernel_size=(win_size_q[0], win_size_q[1]))
        self.avgpool_k = nn.AvgPool2d(kernel_size=(win_size_k[0], win_size_k[1]))
        self.softmax = nn.Softmax(dim=-1)

        self.q_lin = nn.Linear(in_chans2, d_model)
        self.k_lin = nn.Linear(in_chans1, d_model)
        self.v_lin = nn.Linear(in_chans1, d_model)
        self.q_lin2 = nn.Linear(in_chans2, d_model)
        self.k_lin2 = nn.Linear(in_chans1, d_model)
        self.v_lin2 = nn.Linear(in_chans1, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.proj = nn.Sequential(
            nn.Linear(in_chans1, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = Mlp(in_features=d_model, hidden_features=mlp_hidden_dim, drop=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x1, x2, with_pos=True):
        # x1: B, C1, H1, W1   smaller size feature for key, value
        # x2: B, C2, H2, W2   larger size feature for query
        B, C1, H1, W1 = x1.shape
        B, C2, H2, W2 = x2.shape
        x1g = x1
        x2g = x2
        x1 = x1.permute(0, 2, 3, 1).contiguous() # B, H1, W1, C1
        shortcut = self.proj(x1)  # B, H2, W2, d_model
        shortcut = F.interpolate(shortcut.permute(0, 3, 1, 2).contiguous(), size=(H2, W2), mode='bilinear',
                                 align_corners=True)
        shortcut = shortcut.permute(0, 2, 3, 1).contiguous()  # B, H2, W2, C2

        pad_l = pad_t = 0
        pad_r = (self.win_size_k[1] - W1 % self.win_size_k[1]) % self.win_size_k[1]
        pad_b = (self.win_size_k[0] - H1 % self.win_size_k[0]) % self.win_size_k[0]
        x1 = F.pad(x1, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, H1p, W1p, _ = x1.shape
        x2 = x2.permute(0, 2, 3, 1).contiguous()  # B, H2, W2, C2
        pad_l = pad_t = 0
        pad_r = (self.win_size_q[1] - W2 % self.win_size_q[1]) % self.win_size_q[1]
        pad_b = (self.win_size_q[0] - H2 % self.win_size_q[0]) % self.win_size_q[0]
        x2 = F.pad(x2, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, H2p, W2p, _ = x2.shape

        x1 = window_partition(x1, self.win_size_k) # nW*B, window_size, window_size, C1
        x2 = window_partition(x2, self.win_size_q) # nW*B, window_size, window_size, C2

        q = x2.view(-1, self.win_size_q[0] * self.win_size_q[1], C2)  # nWq*B, window_size*window_size, C
        k = x1.view(-1, self.win_size_k[0] * self.win_size_k[1], C1)  # nWq*B, window_size*window_size, C
        # H, W = self.resolution
        B_, N1, C2 = q.shape
        B_, N2, C1 = k.shape
        q = self.q_lin(q).reshape(B_, N1, self.nhead, self.d_model // self.nhead).permute(0, 2, 1, 3).contiguous()
        k2 = self.k_lin(k).reshape(B_, N2, self.nhead, self.d_model // self.nhead).permute(0, 2, 1, 3).contiguous()
        v = self.v_lin(k).reshape(B_, N2, self.nhead, self.d_model // self.nhead).permute(0, 2, 1, 3).contiguous()
        k = k2
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        q = (attn @ v).transpose(1, 2).reshape(B_, N1, self.d_model)
        q = q.view(-1, self.win_size_q[0], self.win_size_q[1], self.d_model)
        # B = int(B_ / (Hp * Wp / self.window_size[0] / self.window_size[1]))
        q = q.view(B, H2p // self.win_size_q[0], W2p // self.win_size_q[1], self.win_size_q[0],
                   self.win_size_q[1], -1)
        q = q.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H2p, W2p, -1)
        # q = shortcut + q # B, H2, W2, d_model
        # shortcut = q

        qg = self.avgpool_q(x2g).permute(0, 2, 3, 1).contiguous().view(B, -1, C2)
        kg = self.avgpool_k(x1g).permute(0, 2, 3, 1).contiguous().view(B, -1, C1)
        num_window_q = qg.shape[1]
        num_window_k = kg.shape[1]
        assert num_window_q==num_window_k
        qg = self.q_lin2(qg).reshape(B, num_window_q, self.nhead, self.d_model // self.nhead).permute(0, 2, 1,
                                                                                                   3).contiguous()
        kg2 = self.k_lin2(kg).reshape(B, num_window_k, self.nhead, self.d_model // self.nhead).permute(0, 2, 1,
                                                                                                    3).contiguous()
        vg = self.v_lin2(kg).reshape(B, num_window_k, self.nhead, self.d_model // self.nhead).permute(0, 2, 1,
                                                                                                   3).contiguous()
        kg = kg2
        qg = qg * self.scale
        attn = (qg @ kg.transpose(-2, -1))
        attn = self.softmax(attn)
        qg = (attn @ vg).transpose(1, 2).reshape(B, num_window_q, self.d_model)
        qg = qg.transpose(1, 2).reshape(B, self.d_model, H2p // self.win_size_q[0], W2p // self.win_size_q[1])
        qg = F.interpolate(qg, size=(H2p, W2p), mode='bilinear', align_corners=True)
        # q = q + qg.permute(0, 2, 3, 1).contiguous() # (B, qHp, qWp, C2)
        qg = qg.permute(0, 2, 3, 1).contiguous()  # (B, qHp, qWp, C2)
        q = q + qg  # B, H2, W2, d_model
        if pad_r > 0 or pad_b > 0:
            q = q[:, :H2, :W2, :].contiguous()
        out = shortcut + self.mlp(self.norm1(q))
        out = self.norm2(out).permute(0, 3, 1, 2).contiguous()
        return out # B, d_model, H2, W2

class TransFusion2(nn.Module):  # Cross-IIWA
    def __init__(self, in_chans1, in_chans2, d_model=128, nhead=8, win_size_k=(12, 12), mlp_ratio=2,
                 win_size_q=(12, 12), dropout=0.1):
        super(TransFusion2, self).__init__()

        self.win_size_q = win_size_q
        self.win_size_k = win_size_k
        self.nhead = nhead
        self.d_model = d_model
        head_dim = d_model // nhead
        self.scale = head_dim ** -0.

        self.pos_enc = PositionEmbeddingSine(normalize=True)
        self.pos_enc2 = PositionEmbeddingSine2(normalize=True)

        self.avgpool_q = nn.Sequential(
            nn.AvgPool2d(kernel_size=(win_size_q[0], win_size_q[1])),
            nn.Conv2d(in_chans2, d_model, 1)
        )
        self.avgpool_k = nn.Sequential(
            nn.AvgPool2d(kernel_size=(win_size_k[0], win_size_k[1])),
            nn.Conv2d(in_chans1, d_model, 1)
        )
        #self.avgpool_q = nn.AvgPool2d(kernel_size=(win_size_q[0], win_size_q[1]))
        #self.avgpool_k = nn.AvgPool2d(kernel_size=(win_size_k[0], win_size_k[1]))
        self.softmax = nn.Softmax(dim=-1)

        self.q_lin = nn.Linear(in_chans2, d_model)
        self.k_lin = nn.Linear(in_chans1, d_model)
        self.v_lin = nn.Linear(in_chans1, d_model)
        self.q_lin2 = nn.Linear(d_model, d_model)
        self.k_lin2 = nn.Linear(d_model, d_model)
        self.v_lin2 = nn.Linear(d_model, d_model)

        self.proj = nn.Sequential(
            nn.Linear(in_chans1, d_model),
            nn.LayerNorm(d_model),
            # nn.GELU()
        )

        self.proj1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

        self.proj2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # mlp_hidden_dim = int(d_model * mlp_ratio)
        # self.mlp = Mlp(in_features=d_model, hidden_features=mlp_hidden_dim, drop=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x1, x2, with_pos=True):
        # x1: B, C1, H1, W1   smaller size feature for key, value
        # x2: B, C2, H2, W2   larger size feature for query
        B, C1, H1, W1 = x1.shape
        B, C2, H2, W2 = x2.shape
        
        x1 = x1.permute(0, 2, 3, 1).contiguous()  # B, H1, W1, C1
        shortcut = self.proj(x1)  # B, H2, W2, d_model
        shortcut = F.interpolate(shortcut.permute(0, 3, 1, 2).contiguous(), size=(H2, W2), mode='bilinear',
                                 align_corners=True)
        shortcut = shortcut.permute(0, 2, 3, 1).contiguous()  # B, H2, W2, C2

        pad_l = pad_t = 0
        pad_r = (self.win_size_k[1] - W1 % self.win_size_k[1]) % self.win_size_k[1]
        pad_b = (self.win_size_k[0] - H1 % self.win_size_k[0]) % self.win_size_k[0]
        x1 = F.pad(x1, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, H1p, W1p, _ = x1.shape
        x1g = x1.permute(0, 3, 1, 2).contiguous()
        x2 = x2.permute(0, 2, 3, 1).contiguous()  # B, H2, W2, C2
        pad_l = pad_t = 0
        pad_r = (self.win_size_q[1] - W2 % self.win_size_q[1]) % self.win_size_q[1]
        pad_b = (self.win_size_q[0] - H2 % self.win_size_q[0]) % self.win_size_q[0]
        x2 = F.pad(x2, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, H2p, W2p, _ = x2.shape
        shortcut_x2 = x2
        # x2g = x2.permute(0, 3, 1, 2).contiguous()

        # x1 = x1 + self.pos_enc(x1)
        # x2 = x2 + self.pos_enc(x2)
        x1 = window_partition(x1, self.win_size_k)  # nW*B, window_size, window_size, C1
        x2 = window_partition(x2, self.win_size_q)  # nW*B, window_size, window_size, C2
        x1 = x1 + self.pos_enc2(x1)
        x2 = x2 + self.pos_enc2(x2)

        q = x2.view(-1, self.win_size_q[0] * self.win_size_q[1], C2)  # nWq*B, window_size*window_size, C
        k = x1.view(-1, self.win_size_k[0] * self.win_size_k[1], C1)  # nWq*B, window_size*window_size, C
        # H, W = self.resolution
        B_, N1, C2 = q.shape
        B_, N2, C1 = k.shape
        q = self.q_lin(q).reshape(B_, N1, self.nhead, self.d_model // self.nhead).permute(0, 2, 1, 3).contiguous()
        k2 = self.k_lin(k).reshape(B_, N2, self.nhead, self.d_model // self.nhead).permute(0, 2, 1, 3).contiguous()
        v = self.v_lin(k).reshape(B_, N2, self.nhead, self.d_model // self.nhead).permute(0, 2, 1, 3).contiguous()
        k = k2
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        q = (attn @ v).transpose(1, 2).reshape(B_, N1, self.d_model)
        q = q.view(-1, self.win_size_q[0], self.win_size_q[1], self.d_model)
        # B = int(B_ / (Hp * Wp / self.window_size[0] / self.window_size[1]))
        q = q.view(B, H2p // self.win_size_q[0], W2p // self.win_size_q[1], self.win_size_q[0],
                   self.win_size_q[1], -1)
        q = q.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H2p, W2p, -1)
        # q = shortcut + q # B, H2, W2, d_model
        # q = shortcut_x2 + self.proj_drop2(self.proj2(q))
        q = shortcut_x2 + self.dropout1(q)
        q = q + self.proj1(q)
        # shortcut = q

        # qg = self.avgpool_q(x2.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        # qg = self.avgpool_q(x2g)
        qg = self.avgpool_q(q.permute(0, 3, 1, 2).contiguous())
        qg = qg + self.pos_enc(qg)
        #print('qg:',qg.shape)
        qg = qg.permute(0, 2, 3, 1).view(B, -1, C2).contiguous()
        # kg = self.avgpool_k(x1.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        kg = self.avgpool_k(x1g)
        kg = kg + self.pos_enc(kg)
        #print('kg:',kg.shape)
        # kg = kg.view(B, -1, C1)
        kg = kg.permute(0, 2, 3, 1).view(B, -1, C2).contiguous()
        
        num_window_q = qg.shape[1]
        num_window_k = kg.shape[1]
        assert num_window_q == num_window_k
        qg = self.q_lin2(qg).reshape(B, num_window_q, self.nhead, self.d_model // self.nhead).permute(0, 2, 1,
                                                                                                      3).contiguous()
        kg2 = self.k_lin2(kg).reshape(B, num_window_k, self.nhead, self.d_model // self.nhead).permute(0, 2, 1,
                                                                                                       3).contiguous()
        vg = self.v_lin2(kg).reshape(B, num_window_k, self.nhead, self.d_model // self.nhead).permute(0, 2, 1,
                                                                                                      3).contiguous()
        kg = kg2
        qg = qg * self.scale
        attn = (qg @ kg.transpose(-2, -1))
        attn = self.softmax(attn)
        qg = (attn @ vg).transpose(1, 2).reshape(B, num_window_q, self.d_model)
        qg = qg.transpose(1, 2).reshape(B, self.d_model, H2p // self.win_size_q[0], W2p // self.win_size_q[1])
        qg = F.interpolate(qg, size=(H2p, W2p), mode='bilinear', align_corners=True)
        # q = q + qg.permute(0, 2, 3, 1).contiguous() # (B, qHp, qWp, C2)
        qg = qg.permute(0, 2, 3, 1).contiguous()  # (B, qHp, qWp, C2)
        # q = q + self.proj_drop3(self.proj3(qg))  # B, H2, W2, d_model
        q = q + self.dropout2(qg)
        q = q + self.proj2(q)
        if pad_r > 0 or pad_b > 0:
            q = q[:, :H2, :W2, :].contiguous()
            # qg = qg[:, :H2, :W2, :].contiguous()
        out = shortcut + self.norm1(q)
        out = self.norm2(out).permute(0, 3, 1, 2).contiguous()
        return out  # B, d_model, H2, W2

class TransFusion3(nn.Module):  # add
    def __init__(self, in_chans1, in_chans2, d_model=128, nhead=8, win_size_k=(12, 12), mlp_ratio=2,
                 win_size_q=(12, 12), dropout=0.1):
        super(TransFusion3, self).__init__()

        self.win_size_q = win_size_q
        self.win_size_k = win_size_k
        self.nhead = nhead
        self.d_model = d_model
        head_dim = d_model // nhead
        self.scale = head_dim ** -0.

        self.pos_enc = PositionEmbeddingSine(normalize=True)

        self.avgpool_q = nn.AvgPool2d(kernel_size=(win_size_q[0], win_size_q[1]))
        self.avgpool_k = nn.AvgPool2d(kernel_size=(win_size_k[0], win_size_k[1]))
        self.softmax = nn.Softmax(dim=-1)

        self.q_lin = nn.Linear(in_chans2, d_model)
        self.k_lin = nn.Linear(in_chans1, d_model)
        self.v_lin = nn.Linear(in_chans1, d_model)
        self.q_lin2 = nn.Linear(in_chans2, d_model)
        self.k_lin2 = nn.Linear(in_chans1, d_model)
        self.v_lin2 = nn.Linear(in_chans1, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.proj = nn.Sequential(
            nn.Linear(in_chans1, d_model),
            nn.LayerNorm(d_model),
            # nn.GELU()
        )

        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = Mlp(in_features=d_model, hidden_features=mlp_hidden_dim, drop=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x1, x2, with_pos=True):
        # x1: B, C1, H1, W1   smaller size feature for key, value
        # x2: B, C2, H2, W2   larger size feature for query
        B, C1, H1, W1 = x1.shape
        B, C2, H2, W2 = x2.shape
        x1 = x1.permute(0, 2, 3, 1).contiguous() # B, H1, W1, C1
        shortcut = self.proj(x1)  # B, H2, W2, d_model
        shortcut = F.interpolate(shortcut.permute(0, 3, 1, 2).contiguous(), size=(H2, W2), mode='bilinear',
                                 align_corners=True)
        shortcut = shortcut.permute(0, 2, 3, 1).contiguous()  # B, H2, W2, C2

        pad_l = pad_t = 0
        pad_r = (self.win_size_k[1] - W1 % self.win_size_k[1]) % self.win_size_k[1]
        pad_b = (self.win_size_k[0] - H1 % self.win_size_k[0]) % self.win_size_k[0]
        x1 = F.pad(x1, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, H1p, W1p, _ = x1.shape
        x2 = x2.permute(0, 2, 3, 1).contiguous()  # B, H2, W2, C2
        pad_l = pad_t = 0
        pad_r = (self.win_size_q[1] - W2 % self.win_size_q[1]) % self.win_size_q[1]
        pad_b = (self.win_size_q[0] - H2 % self.win_size_q[0]) % self.win_size_q[0]
        x2 = F.pad(x2, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, H2p, W2p, _ = x2.shape
        shortcut_x2 = x2

        # x1 = x1 + self.pos_enc(x1)
        # x2 = x2 + self.pos_enc(x2)
        x1 = window_partition(x1, self.win_size_k) # nW*B, window_size, window_size, C1
        x2 = window_partition(x2, self.win_size_q) # nW*B, window_size, window_size, C2
        # x1 = x1 + self.pos_enc(x1)
        # x2 = x2 + self.pos_enc(x2)

        q = x2.view(-1, self.win_size_q[0] * self.win_size_q[1], C2)  # nWq*B, window_size*window_size, C
        k = x1.view(-1, self.win_size_k[0] * self.win_size_k[1], C1)  # nWq*B, window_size*window_size, C
        # H, W = self.resolution
        B_, N1, C2 = q.shape
        B_, N2, C1 = k.shape
        q = self.q_lin(q).reshape(B_, N1, self.nhead, self.d_model // self.nhead).permute(0, 2, 1, 3).contiguous()
        k2 = self.k_lin(k).reshape(B_, N2, self.nhead, self.d_model // self.nhead).permute(0, 2, 1, 3).contiguous()
        v = self.v_lin(k).reshape(B_, N2, self.nhead, self.d_model // self.nhead).permute(0, 2, 1, 3).contiguous()
        k = k2
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        q = (attn @ v).transpose(1, 2).reshape(B_, N1, self.d_model)
        q = q.view(-1, self.win_size_q[0], self.win_size_q[1], self.d_model)
        # B = int(B_ / (Hp * Wp / self.window_size[0] / self.window_size[1]))
        q = q.view(B, H2p // self.win_size_q[0], W2p // self.win_size_q[1], self.win_size_q[0],
                   self.win_size_q[1], -1)
        q = q.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H2p, W2p, -1)
        # q = shortcut + q # B, H2, W2, d_model
        # q = shortcut_x2 + self.proj_drop2(self.proj2(q))
        q = shortcut_x2 + q
        # shortcut = q

        qg = self.avgpool_q(x2.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        qg = qg + self.pos_enc(qg)
        qg = qg.view(B, -1, C2)
        kg = self.avgpool_k(x1.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        kg = kg + self.pos_enc(kg)
        kg = kg.view(B, -1, C1)
        num_window_q = qg.shape[1]
        num_window_k = kg.shape[1]
        assert num_window_q==num_window_k
        qg = self.q_lin2(qg).reshape(B, num_window_q, self.nhead, self.d_model // self.nhead).permute(0, 2, 1,
                                                                                                   3).contiguous()
        kg2 = self.k_lin2(kg).reshape(B, num_window_k, self.nhead, self.d_model // self.nhead).permute(0, 2, 1,
                                                                                                    3).contiguous()
        vg = self.v_lin2(kg).reshape(B, num_window_k, self.nhead, self.d_model // self.nhead).permute(0, 2, 1,
                                                                                                   3).contiguous()
        kg = kg2
        qg = qg * self.scale
        attn = (qg @ kg.transpose(-2, -1))
        attn = self.softmax(attn)
        qg = (attn @ vg).transpose(1, 2).reshape(B, num_window_q, self.d_model)
        qg = qg.transpose(1, 2).reshape(B, self.d_model, H2p // self.win_size_q[0], W2p // self.win_size_q[1])
        qg = F.interpolate(qg, size=(H2p, W2p), mode='bilinear', align_corners=True)
        # q = q + qg.permute(0, 2, 3, 1).contiguous() # (B, qHp, qWp, C2)
        qg = qg.permute(0, 2, 3, 1).contiguous()  # (B, qHp, qWp, C2)
        # q = q + self.proj_drop3(self.proj3(qg))  # B, H2, W2, d_model
        q = q + qg
        if pad_r > 0 or pad_b > 0:
            q = q[:, :H2, :W2, :].contiguous()
            # qg = qg[:, :H2, :W2, :].contiguous()
        # out = shortcut + self.mlp(self.norm1(q))
        # out = shortcut + self.norm1(q)
        # q = torch.cat([q, qg], dim=-1)
        # out = shortcut + self.proj2(q)
        out = shortcut + self.mlp(self.norm1(q))
        out = self.norm2(out).permute(0, 3, 1, 2).contiguous()
        return out # B, d_model, H2, W2

@NECKS.register_module()
class Cascade_TRP(nn.Module):

    def __init__(self,
                 in_channels=[], out_channels=512, d_models=[], n_head=[8, 8, 8],
                 dim_feedforwards=[], patch_sizes=[[1,1], [1,1], [1,1], [1,1], [1,1], [1,1],],
                 dropout_ratio=0.1,
                 num_outs=3,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 out_type='1',
                 overlap=False,
                 cascade_num=1,
                 pos_type='sin',
                 upsample_cfg=dict(mode='nearest'),
                 adapt_size1=(10, 10),
                 adapt_size2=(10, 10),
                 extra_out=False):
        super(Cascade_TRP, self).__init__()
        # to do

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

if __name__ == '__main__':
    x = [torch.rand(1, 96, 56, 56), torch.rand(1, 192, 28, 28), torch.rand(1, 384, 28, 28), torch.rand(1, 768, 28, 28)]
    trp = Cascade_TRP(
        in_channels=[96, 192, 384, 768],
        out_channels=512,
        d_models=[96, 192, 384],
        n_head=[8, 8, 8],
        dim_feedforwards=[384, 768, 1536],
        patch_sizes=[[2,2], [4,4], [2,2], [2,2], [2,2], [2,2],],
        num_outs=3,
        cascade_num=1,
        adapt_size1=(7,7),
        adapt_size2=(7,7),
    )
    y = trp(x)
    pass

# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import torch
import torch.nn as nn
from einops import rearrange

class MTAttention(nn.Module):
    def __init__(self, dim, qkv_bias=False):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([dim**-0.5]), requires_grad=False)

        self.attend = nn.Softmax(dim=-1)
        self.to_q_ = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim, bias=qkv_bias),
            nn.GELU(),
            nn.Linear(dim, dim, bias=qkv_bias),
            nn.GELU(),
            nn.Linear(dim, dim, bias=qkv_bias),
        )
        self.to_k_ = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim, bias=qkv_bias),
            nn.GELU(),
            nn.Linear(dim, dim, bias=qkv_bias),
            nn.GELU(),
            nn.Linear(dim, dim, bias=qkv_bias),
        )
    
    def forward(self, query, key, only_attn=False):
        B, N, T, C, H, W = key.shape
        query, key = map(lambda t: rearrange(t, 'b n t c h w -> (b t h w) n c'), [query, key])
        value = key
        assert T == 3
        q_ = self.to_q_(query); k_ = self.to_k_(key)
        dots = torch.matmul(q_, k_.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        if only_attn:
            attn = rearrange(attn, '(b t h w) n c -> b n t c h w', b=B, t=T, h=H, w=W)
            return attn[:, 0]
        value = torch.matmul(attn, value)
        value = rearrange(value, '(b t h w) n c -> b n t c h w', b=B, t=T, h=H, w=W)
        assert value.shape[1] == 1
        return value[:, 0]


class TPTransformer(nn.Module):
    def __init__(self, dim, depth=3, num_heads=4, qkv_bias=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias),
                FeedForward(dim)
            ]))
    
    def forward(self, x):
        B, N, T, C, H, W = x.shape
        assert T == 3
        x = rearrange(x, 'b n t c h w -> (b t h w) n c')
        # x: [4*3*65536, 3, 32]
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = self.norm(x)
        x = rearrange(x, '(b t h w) n c -> b n t c h w', b=B, t=T, h=H, w=W)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=True):
        super().__init__()
        self.heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.to_out = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

if __name__ == '__main__':
    x = torch.randn(4, 3, 3, 32, 32)
    query = torch.randn(4, 1, 3, 32, 256, 256).cuda(7)
    key = torch.randn(4, 2, 3, 32, 256, 256).cuda(7)
    model = SoftTPTransformer(32, depth=2).cuda(7)
    y = model(query, key)
    print(y.shape)

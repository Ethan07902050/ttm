import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from module import Attention, PreNorm, FeedForward
import numpy as np

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

  
class ViViT_w_Audio_v1(nn.Module):
    def __init__(self, image_size_H, image_size_W, patch_size_h, patch_size_w, audio_dim, max_num_frames, dim = 256, depth = 6, heads = 4, pool = 'cls', in_channels = 3, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., scale_dim = 4, audio_scale = None):
        super().__init__()

        """ base on vivit from: https://github.com/rishikksh20/ViViT-pytorch/blob/master/vivit.py """
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        assert image_size_H % patch_size_h == 0, 'Image dimensions must be divisible by the patch size.'
        assert image_size_W % patch_size_w == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_size_H // patch_size_h) * (image_size_W // patch_size_w) # 64
        patch_dim = patch_size_h * patch_size_w * in_channels

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size_h, p2 = patch_size_w),
            nn.Linear(patch_dim, dim),
        )

        self.audio_net = nn.Sequential(
            nn.Linear(audio_dim, audio_dim),
            nn.Linear(audio_dim, dim)
        )

        self.img_pos_embedding = nn.Parameter(torch.randn(1, max_num_frames, num_patches + 2, dim)) # add cls token & audio imformation

        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1), # for binary classification
            nn.Sigmoid()
        )

    def forward(self, x, audio):
        
        # patch embeddings
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        # add cls token
        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t)

        # add audio imformation
        audio = self.audio_net(audio)
        x = torch.cat((cls_space_tokens, audio.unsqueeze(2), x), dim=2)

        # position embeddings
        x += self.img_pos_embedding[:, :(t), :(n + 2)]

        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)
        
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)

class ViViT_w_Audio_v2(nn.Module):
    def __init__(self, image_size_H, image_size_W, patch_size_h, patch_size_w, 
                audio_dim, max_num_frames, dim = 256, depth = 6, heads = 4, 
                pool = 'cls', in_channels = 3, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., scale_dim = 4, audio_scale = 6):
        super().__init__()

        """ base on vivit from: https://github.com/rishikksh20/ViViT-pytorch/blob/master/vivit.py """
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        assert image_size_H % patch_size_h == 0, 'Image dimensions must be divisible by the patch size.'
        assert image_size_W % patch_size_w == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_size_H // patch_size_h) * (image_size_W // patch_size_w) # 64
        patch_dim = patch_size_h * patch_size_w * in_channels

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size_h, p2 = patch_size_w),
            nn.Linear(patch_dim, dim),
        )

        self.audio_net = nn.Sequential(
            nn.Linear(audio_dim, audio_dim*audio_scale),
            nn.Linear(audio_dim*audio_scale, audio_dim*audio_scale),
            nn.Linear(audio_dim*audio_scale, dim)
        )

        self.audio_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.img_pos_embedding = nn.Parameter(torch.randn(1, max_num_frames, num_patches + 2, dim)) # add cls token & audio imformation

        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1), # for binary classification
            nn.Sigmoid()
        )

    def forward(self, x, audio):
        
        # patch embeddings
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        # add cls token
        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t)

        # add audio imformation
        audio = self.audio_net(audio)
        x = torch.cat((cls_space_tokens, audio.unsqueeze(2), x), dim=2)
        # position embeddings
        x += self.img_pos_embedding[:, :(t), :(n + 2)]

        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        # audio transformer
        audio = self.audio_transformer(audio)[:, 0]

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, audio.unsqueeze(1), x), dim=1)
        x = self.temporal_transformer(x)
        
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = 32
    Q = 128
    img = torch.ones([B, Q, 3, 96, 96]).cuda()
    audio = torch.ones([B, Q, 64]).cuda()
    
    model = ViViT_w_Audio_v1(
        image_size_H=96,
        image_size_W=96,
        patch_size_h=32,
        patch_size_w=32,
        max_num_frames=Q,
        in_channels=3,
        audio_dim=64,
        dim=256,
        depth=8
    ).cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
    # for i in range(20000):
    out = model(img, audio)
    
    print("Shape of out :", out.shape)      # [B, num_classes]

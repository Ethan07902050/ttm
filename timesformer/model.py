import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from module import Attention, PreNorm, FeedForward
import numpy as np
from transformers import Wav2Vec2Processor, HubertForCTC, HubertModel, Wav2Vec2Model
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
from datasets import load_dataset
import librosa

from timesformer.models.vit import TimeSformer
import copy
import timm
from typing import Optional, List

class AudioStream(nn.Module):
    def __init__(self, output_layer=9, hidden_dim=1024, output_dim=768, audio_replacement=None):
        super().__init__()
        
        self.output_layer = output_layer
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.audio_replacement = audio_replacement

        self.extractor = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h", mask_time_length=2, cache_dir="/tmp2/b08902028/cache")
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2), 
        )
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, audio):
        try:
            x = self.extractor(**audio).last_hidden_state
        except:
            x = self.extractor(audio['input_values']).last_hidden_state
        # print("x shape: ", x.shape)
        # x = x[self.output_layer]

        print(x.shape)

        return self.mlp_head(x)

class VideoStream(nn.Module):
    def __init__(self, img_size=96, num_classes=2, num_frames=128, video_cls_token_num=1):
        super().__init__()

        # self.activation = {}
        # def get_activation(name):
        #     def hook(model, input, output):
        #         self.activation[name] = output.detach()
        #     return hook

        self.model = TimeSformer(img_size=img_size, 
                            num_classes=num_classes, 
                            num_frames=num_frames, 
                            attention_type='divided_space_time',  
                            video_cls_token_num=video_cls_token_num,
                            pretrained_model='/tmp2/b08902028/DLCV/TimeSformer/checkpoints/TimeSformer_divST_96x4_224_K600.pyth')
        # self.model.model.blocks[-3].register_forward_hook(get_activation('feats'))

        self.video_cls_token_num = video_cls_token_num
        self.video_cls_token = nn.Parameter(torch.randn(1, 3, 96, 96))

    def forward(self, clip):
        # b_size = clip.shape[0]

        # video_cls_tokens = repeat(self.video_cls_token, '() c h w  -> b t c h w', b = b_size, t=self.video_cls_token_num)
        # concat_clip = torch.cat((video_cls_tokens, clip), dim=1) # output: [B, N, 768]

        return self.model(clip)

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, 
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        # print(tgt.shape)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(tgt, pos),
                                   value=tgt, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        # print(tgt2.shape)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        return self.forward_post(tgt, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class TTM(nn.Module):
    def __init__(self, output_layer=9, hidden_dim=768, output_dim=768,
                img_size=96, num_classes=2, num_frames=128, d_model=768, nhead=4,
                num_decoder_layers=2, dim_feedforward=2048, dropout=0.1,
                activation="relu", device=None, cls_token_num=1, audio_replacement = None):
        super().__init__()

        self.audio_stream = AudioStream(output_layer, hidden_dim, output_dim, audio_replacement=audio_replacement)
        self.video_stream = VideoStream(img_size, num_classes, num_frames)

        # for the decoder
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.cls_token = nn.Parameter(torch.randn(1, output_dim))
        self.cls_token_num = cls_token_num

        self.cls_mlp = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, num_classes), # for binary classification
        )

    def forward(self, clip, audio):

        b_size = clip.shape[0]
        clip_x = self.video_stream(clip)
        # print(audio)
        audio_x = self.audio_stream(audio)

        # concatentate the tokens 
        cls_space_tokens = repeat(self.cls_token, '() d -> b t d', b = b_size, t=self.cls_token_num)
        concat_x = torch.cat((cls_space_tokens, clip_x, audio_x), dim=1) # output: [B, N, 768]

        # multihead attention blocks
        x = self.decoder(concat_x)
        
        # mean the cls_tokens and go through mlp layer
        cls_tokens = x[:, :self.cls_token_num, :]
        cls_tokens = torch.squeeze(cls_tokens)
        # cls_tokens = torch.mean(cls_tokens, axis=1)
        x = torch.sigmoid(self.cls_mlp(cls_tokens))

        return x

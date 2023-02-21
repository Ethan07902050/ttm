import torch
import torch.nn as nn
import torch.nn.functional as F
from model.audioEncoder      import audioEncoder
from model.visualEncoder     import visualFrontend, visualTCN, visualConv1D
from model.attentionLayer    import attentionLayer

class talkNetModel(nn.Module):
    def __init__(self):
        super(talkNetModel, self).__init__()
        # Visual Temporal Encoder
        self.visualFrontend  = visualFrontend() # Visual Frontend 
        # self.visualFrontend.load_state_dict(torch.load('visual_frontend.pt', map_location="cuda"))
        # for param in self.visualFrontend.parameters():
        #     param.requires_grad = False       
        self.visualTCN       = visualTCN()      # Visual Temporal Network TCN
        self.visualConv1D    = visualConv1D()   # Visual Temporal Network Conv1d

        # Audio Temporal Encoder 
        self.audioEncoder  = audioEncoder(layers = [3, 4, 6, 3],  num_filters = [16, 32, 64, 128])
        
        # Audio-visual Cross Attention
        self.crossA2V = attentionLayer(d_model = 128, nhead = 8)
        self.crossV2A = attentionLayer(d_model = 128, nhead = 8)

        # Audio-visual Self Attention
        self.selfAV = attentionLayer(d_model = 256, nhead = 8)
        self.fcAV = nn.Linear(256, 2)
        self.fcA = nn.Linear(128, 2)
        self.fcV = nn.Linear(128, 2)

    def forward_visual_frontend(self, x):
        B, T, W, H = x.shape  
        x = x.view(B*T, 1, 1, W, H)
        x = (x / 255 - 0.4161) / 0.1688
        x = self.visualFrontend(x)
        x = x.view(B, T, 512)        
        x = x.transpose(1,2)     
        x = self.visualTCN(x)
        x = self.visualConv1D(x)
        x = x.transpose(1,2)
        return x

    def forward_audio_frontend(self, x):    
        x = x.unsqueeze(1).transpose(2, 3)        
        x = self.audioEncoder(x)
        return x

    def forward_cross_attention(self, audio, video):
        audio_c, w_v_2_a = self.crossA2V(src = audio, tar = video)
        video_c, w_a_2_v = self.crossV2A(src = video, tar = audio)
        w_a = torch.sum(w_a_2_v, dim=-1)
        w_a = F.softmax(w_a, dim=-1)
        return audio_c, video_c, w_a

    def forward_audio_visual_backend(self, x1, x2, w=None): 
        x = torch.cat((x1,x2), 2)    
        x, w_s = self.selfAV(src = x, tar = x)
        if w != None:
            x = torch.einsum('bte,bt->be', x, w)
        else:
            x = torch.mean(x, dim=1)
            # w_s = torch.sum(w_s, dim=-1)
            # w_s = F.softmax(w_s, dim=-1)
            # x = torch.einsum('bte,bt->be', x, w_s)
        # x = torch.reshape(x, (-1, 256))
        x = x.squeeze(1)
        x = self.fcAV(x)
        return x    

    def forward_audio_backend(self, x, w=None):
        if w != None:
            x = torch.einsum('bte,bt->be', x, w)
        else:
            x = torch.mean(x, dim=1)
        # x = torch.reshape(x, (-1, 128))
        x = x.squeeze(1)
        x = self.fcA(x)
        return x

    def forward_visual_backend(self,x, w=None):
        if w != None:
            x = torch.einsum('bte,bt->be', x, w)
        else:
            x = torch.mean(x, dim=1)
        # x = torch.reshape(x, (-1, 128))
        x = x.squeeze(1)
        x = self.fcV(x)
        return x


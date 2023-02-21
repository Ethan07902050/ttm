from torchvision import models
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import torch.nn as nn
import torch

class AudioEncoder(nn.Module):
    def __init__(self, model_name):
        super(AudioEncoder, self).__init__()
        self.model = HubertModel.from_pretrained(model_name)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    
    def forward(self, audio, audio_attention, audio_len):
        # print(audio)
        # print(audio_attention)
        inputs = self.feature_extractor(audio, sampling_rate=16000, padding=True, return_tensors="pt", return_attention_mask=True)
        output = self.model(
            input_values=inputs['input_values'].cuda(),
            attention_mask=inputs['attention_mask'].cuda(),
            output_hidden_states=True,
        )
        # output = self.model(input_values=audio, attention_mask=audio_attention)
        # last_hidden = output['last_hidden_state']
        last_hidden = output['hidden_states'][7]
        pooled_output = []
        for i in range(len(audio_len)):
            if audio_len[i] > 0:
                pooled_output.append(last_hidden[i][:audio_len[i]].mean(dim=0))
            else:
                pooled_output.append(last_hidden[i].mean(dim=0))
        pooled_output = torch.stack(pooled_output)
        return pooled_output

class VideoEncoder(nn.Module):
    def __init__(self):
        super(VideoEncoder, self).__init__()
        self.model = models.video.r2plus1d_18(weights="R2Plus1D_18_Weights.KINETICS400_V1")
        self.model.fc = nn.Identity()
    def forward(self, video):
        return self.model(video)

class Classifier(nn.Module):
    def __init__(self, config) -> None:
        super(Classifier, self).__init__()
        self.audio_encoder = AudioEncoder(config['audio_encoder'])
        self.video_encoder = VideoEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(512+768, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1, bias=True),
            nn.Sigmoid(),
        )
        self.classifier_a = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1, bias=True),
            nn.Sigmoid(),
        )
        self.classifier_v = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, batch):
        video = batch['video'].cuda()
        audio = batch['audio']
        audio_len = batch['audio_length']
        audio_attention_mask = batch['audio_attention_mask'].cuda()
        video_feature = self.video_encoder(video)
        audio_feature = self.audio_encoder(audio, audio_attention_mask, audio_len)
        concated_feature = torch.concat([video_feature, audio_feature], dim=-1)
        return self.classifier(concated_feature), self.classifier_a(audio_feature), self.classifier_v(video_feature)


class ClassifierOnlyVideo(nn.Module):
    def __init__(self, config) -> None:
        super(ClassifierOnlyVideo, self).__init__()
        self.video_encoder = VideoEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, batch):
        video = batch['video'].cuda()
        video_feature = self.video_encoder(video)
        return self.classifier(video_feature)



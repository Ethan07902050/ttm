import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TTMDatasetTimesformer
import torchvision.transforms as transforms
import argparse
from model import *
from tqdm import tqdm
from custom_transform import *
from transformers import VideoMAEFeatureExtractor, VideoMAEImageProcessor, TimesformerForVideoClassification

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--continue_weight', default='', type=str)
    return parser

parser = get_parser()
args = parser.parse_args()

config = {
    "batch_size" : 2,
    "lr" : 1e-4,
    "epochs" : 10,
    "audio_encoder" : "facebook/hubert-base-ls960",
    "output_dir" : args.output_dir,
    "audio_length_limit" : 5,
    "video_length_limit" : 24,
}

total_data = [os.path.join(args.train_data, x) for x in sorted(os.listdir(args.train_data), key = lambda x:(len(x), x)) if x.endswith('.npz')]
train_data = total_data[int(len(total_data)*0.05):int(len(total_data)*0.95)]
valid_data = total_data[int(len(total_data)*0.95):] + total_data[:int(len(total_data)*0.05)]


tfm = transforms.Compose([
    RandomHorizontalFlipVideo(),
])

train_dataset = TTMDatasetTimesformer(train_data, config, None, 'train')
valid_dataset = TTMDatasetTimesformer(valid_data, config, None, 'valid')

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=config["batch_size"], 
    shuffle=True,
    collate_fn=train_dataset.collate
)

valid_dataloader = DataLoader(
    valid_dataset, 
    batch_size=config["batch_size"], 
    shuffle=False,
    collate_fn=valid_dataset.collate
)
feature_extractor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
# feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")
model.classifier = nn.Linear(768, 2)
model.cuda()

if len(args.continue_weight):
    model.load_state_dict(torch.load(args.continue_weight), strict=False)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-4, amsgrad=True)
criterion = torch.nn.CrossEntropyLoss()
num_step = 0
best_acc = 0
for epoch in range(config['epochs']):
    epoch_loss = 0.0
    progress_bar = tqdm(range(len(train_dataloader)))
    train_iter = iter(train_dataloader)
    progress_bar.set_description(f"Epoch {epoch}")
    model.train()
    for i in progress_bar:
        # try:
        data = next(train_iter)
        # except:
        #     continue
        inputs = feature_extractor(data['video'], return_tensors="pt", padding=True)
        # print(inputs['pixel_values'].shape)
        outputs = model(pixel_values=inputs['pixel_values'].cuda())
        logits = outputs.logits.squeeze(-1)
        # print(logits)
        data['label'] = data['label'].cuda()
        loss = criterion(logits, data['label'])
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        progress_bar.set_postfix(loss=loss.item())
        num_step += 1

    print(f'epoch {epoch} train loss : {epoch_loss / len(train_dataloader)}')
    model.eval()
    
    total_acc = 0
    total_loss = 0
    total_num = 0
    
    with torch.no_grad():
        valid_iter = iter(valid_dataloader)
        for i in tqdm(range(len(valid_dataloader))):
            # try:
            data = next(valid_iter)
            # except:
            #     continue
            inputs = feature_extractor(data['video'], return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
            data['label'] = data['label'].cuda()
            loss = criterion(logits, data['label'])
            total_loss += loss.item()

            predict = logits.argmax(dim=-1)
            acc = (predict == data['label'].cuda()).sum()
            total_acc += acc
            total_num += len(data['label'])

        print(f"Epoch {epoch} val acc : {total_acc / total_num}, val loss : {total_loss / len(valid_dataloader)}")     
        if (total_acc / total_num > best_acc):
            best_acc = total_acc / total_num
            print(f"Save model with best acc : {total_acc / total_num}")
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), os.path.join(args.output_dir, f'model_best.pth'))
            else: 
                torch.save(model.state_dict(), os.path.join(args.output_dir, f'model_best.pth'))

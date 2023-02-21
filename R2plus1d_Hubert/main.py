import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TTMDataset
import torchvision.transforms as transforms
import argparse
from model import *
from tqdm import tqdm
from custom_transform import *

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--continue_weight', default='', type=str)
    return parser

parser = get_parser()
args = parser.parse_args()

config = {
    "batch_size" : 4,
    "lr" : 1e-4,
    "epochs" : 10,
    "audio_encoder" : "facebook/hubert-base-ls960",
    "output_dir" : args.output_dir,
    "audio_length_limit" : 5,
    "video_length_limit" : 96,
}

total_data = [os.path.join(args.train_data, x) for x in sorted(os.listdir(args.train_data), key = lambda x:(len(x), x)) if x.endswith('.npz')]
train_data = total_data[:int(len(total_data)*0.9)]
valid_data = total_data[int(len(total_data)*0.9):]

tfm = transforms.Compose([
    RandomHorizontalFlipVideo(),
])

train_dataset = TTMDataset(train_data, config, tfm, 'train')
valid_dataset = TTMDataset(valid_data, config, None, 'valid')

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
model = Classifier(config).cuda()
if len(args.continue_weight):
    model.load_state_dict(torch.load(args.continue_weight), strict=False)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
criterion = torch.nn.BCELoss()
num_step = 0
best_acc = 0
for epoch in range(config['epochs']):
    epoch_loss = 0.0
    progress_bar = tqdm(range(len(train_dataloader)))
    train_iter = iter(train_dataloader)
    progress_bar.set_description(f"Epoch {epoch}")
    model.train()
    for i in progress_bar:
        data = next(train_iter)
        logits, logits_a, logits_v = model(data)
        logits = logits.squeeze(-1)
        logits_a = logits_a.squeeze(-1)
        logits_v = logits_v.squeeze(-1)
        data['label'] = data['label'].cuda()
        loss = criterion(logits, data['label'])
        loss_a = criterion(logits_a, data['label'])
        loss_v = criterion(logits_v, data['label'])
        weight_loss = loss + 0.2 * loss_a + 0.2 * loss_v
        epoch_loss += weight_loss.item()
        optimizer.zero_grad()
        weight_loss.backward()
        optimizer.step()
        
        progress_bar.set_postfix(loss=weight_loss.item())
        num_step += 1

    print(f'epoch {epoch} train loss : {epoch_loss / len(train_dataloader)}')
    model.eval()
    
    total_acc = 0
    total_loss = 0
    total_num = 0
    
    with torch.no_grad():
        valid_iter = iter(valid_dataloader)
        for i in tqdm(range(len(valid_dataloader))):
            data = next(valid_iter)
            logits, logits_a, logits_v = model(data)
            logits = logits.squeeze(-1)
            logits_a = logits_a.squeeze(-1)
            logits_v = logits_v.squeeze(-1)
            data['label'] = data['label'].cuda()
            loss = criterion(logits, data['label'])
            loss_a = criterion(logits_a, data['label'])
            loss_v = criterion(logits_v, data['label'])
            weight_loss = loss + 0.2 * loss_a + 0.2 * loss_v
            total_loss += weight_loss.item()

            predict = (logits > 0.5).float()
            acc = (predict == data['label'].cuda()).sum()
            total_acc += acc
            total_num += len(data['label'])

        print(f"Epoch {epoch} val acc : {total_acc / total_num}, val loss : {total_loss / len(valid_dataloader)}")     
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), os.path.join(args.output_dir, f'model_{epoch}.pth'))
        else: 
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'model_{epoch}.pth'))

        if (total_acc / total_num > best_acc):
            best_acc = total_acc / total_num
            print(f"Save model with best acc : {total_acc / total_num}")
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), os.path.join(args.output_dir, f'model_best.pth'))
            else: 
                torch.save(model.state_dict(), os.path.join(args.output_dir, f'model_best.pth'))

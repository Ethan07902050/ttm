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
import csv
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_data', type=str)
    parser.add_argument('model_dir', type=str)
    return parser

parser = get_parser()
args = parser.parse_args()

config = {
    "batch_size" : 4,
    "lr" : 3e-5,
    "epochs" : 10,
    "audio_encoder" : "superb/hubert-base-superb-sid",
    "audio_length_limit" : 8,
    "video_length_limit" : 128,
}

test_data = [os.path.join(args.test_data, x) for x in sorted(os.listdir(args.test_data), key = lambda x:(len(x), x)) if x.endswith('.npz')]
test_dataset = TTMDataset(test_data[:int(len(test_data)*0.1)], config, None, 'valid')

valid_dataloader = DataLoader(
    test_dataset, 
    batch_size=config["batch_size"], 
    shuffle=False,
    collate_fn=test_dataset.collate
)
model = Classifier(config).cuda()
model.load_state_dict(torch.load(args.model_dir))
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.eval()

total_acc = 0
only_v_acc = 0
only_a_acc = 0
merge_acc = 0
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
        
        predict = (logits > 0.5).float()
        acc = (predict == data['label'].cuda()).sum()
        total_acc += acc
        total_num += len(data['label'])

        predict = (logits_v > 0.5).float()
        only_v_acc += (predict == data['label'].cuda()).sum()

        predict = (logits_a > 0.5).float()
        only_a_acc += (predict == data['label'].cuda()).sum()

        predict = ((10/12*logits + 2/12 * logits_v) > 0.5).float()
        merge_acc += (predict == data['label'].cuda()).sum()
    print(f"val acc : {total_acc / total_num}")
    print(f"only v acc : {only_v_acc / total_num}")    
    print(f"only a acc : {only_a_acc / total_num}")    
    print(f"merge acc : {merge_acc / total_num}")    


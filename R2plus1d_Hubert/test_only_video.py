import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TTMDatasetOnlyVideo
import torchvision.transforms as transforms
import argparse
from model import *
from tqdm import tqdm
import csv
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--model_dir', type=str)
    return parser

parser = get_parser()
args = parser.parse_args()

config = {
    "batch_size" : 4,
    "lr" : 3e-5,
    "epochs" : 10,
    "output_dir" : args.output_dir,
    "audio_length_limit" : 8,
    "video_length_limit" : 128,
}

test_data = [os.path.join(args.test_data, x) for x in sorted(os.listdir(args.test_data), key = lambda x:(len(x), x)) if x.endswith('.npz')]
test_dataset = TTMDatasetOnlyVideo(test_data, config, None, 'test')

test_dataloader = DataLoader(
    test_dataset, 
    batch_size=config["batch_size"], 
    shuffle=False,
    collate_fn=test_dataset.collate
)
model = ClassifierOnlyVideo(config).cuda()
model.load_state_dict(torch.load(args.model_dir))
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.eval()
predicts = {}
with torch.no_grad():
    test_iter = iter(test_dataloader)
    for _ in tqdm(range(len(test_dataloader))):
        data = next(test_iter)
        out = model(data)
        out = out.squeeze(-1)
        predict = (out > 0.5).float()
        for i in range(len(predict)):
            predicts[str(data['id'][i])] = int(predict[i].item())

with open(args.output_dir, 'w') as f:
    f.write('Id,Predicted\n')
    for key, value in predicts.items():
        f.write(f'{key},{value}\n')
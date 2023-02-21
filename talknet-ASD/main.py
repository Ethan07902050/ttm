import os, torch, argparse
from dataLoader import train_loader, val_loader
from talkNet import talkNet
import numpy as np
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--continue_weight', default='', type=str)
    return parser

parser = get_parser()
args = parser.parse_args()

config = {
    "batch_size" : 1600,
    "lr" : 0.0001,
    "lr_decay" : 0.95,
    "weighted" : False,
    "audio_aug" : False,
    "epochs" : 10,
    "output_dir" : args.output_dir,
}

total_data = [os.path.join(args.train_data, x) for x in sorted(os.listdir(args.train_data), key = lambda x:(len(x), x)) if x.endswith('.npz')]
# total_data = total_data[:100]


path_to_len = {}
print("loading data")
for i, path in tqdm(enumerate(total_data)):
    data = np.load(path)
    length = len(data['is_empty'])
    path_to_len[path] = length

train_data = total_data[int(len(total_data)*0.05):int(len(total_data)*0.95)]
valid_data = total_data[int(len(total_data)*0.95):] + total_data[:int(len(total_data)*0.05)]

train_data = sorted(train_data, key=lambda file: path_to_len[file], reverse=True)

train_dataset = train_loader(train_data, config['batch_size'], path_to_len, config['audio_aug'])
trainLoader = torch.utils.data.DataLoader(train_dataset, batch_size = 1, shuffle = True, num_workers = 2)
valid_dataset = val_loader(valid_data)
valLoader = torch.utils.data.DataLoader(valid_dataset, batch_size = 1, shuffle = False, num_workers = 2)

trainer = talkNet(outputPath=config['output_dir'], lr=config['lr'], lrDecay=config['lr_decay'], weighted=config['weighted'])
for epoch in range(1, config["epochs"]+1):        
    loss, lr = trainer.train_network(epoch=epoch, loader=trainLoader)
    trainer.valid_network(valLoader)
    
    

    
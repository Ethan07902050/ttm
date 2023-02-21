import os, torch, argparse
from dataLoader import test_loader
from utils.tools import *
from talkNet import talkNet
import numpy as np
import random
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
    "batch_size" : 1600,
    "lr" : 0.0001,
    "lr_decay" : 0.95,
    "weighted" : False,
    "epochs" : 25,
    "output_dir" : args.output_dir,
}

test_data = [os.path.join(args.test_data, x) for x in sorted(os.listdir(args.test_data), key = lambda x:(len(x), x)) if x.endswith('.npz')]
print("loading data")

# for i, path in tqdm(enumerate(total_data)):
#     data = np.load(path)
#     length = len(data['is_empty'])
#     path_to_len[path] = length

# total_data = sorted(total_data, key=lambda file: path_to_len[file], reverse=True)
# end = len(total_data)-1
# for i in range(len(total_data)-1, -1, -1):
#     if path_to_len[total_data[i]] != 0:
#         end = i
#         break
# total_data = total_data[:end+1]
# random.shuffle(total_data)
# train_data = total_data[:int(len(total_data)*0.9)]
# valid_data = total_data[int(len(total_data)*0.9):]

# train_dataset = train_loader(train_data, config['batch_size'], path_to_len)
# trainLoader = torch.utils.data.DataLoader(train_dataset, batch_size = 1, shuffle = True, num_workers = 2)
test_dataset = test_loader(test_data)
testLoader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False, num_workers = 2)

trainer = talkNet(outputPath=config['output_dir'], lr=config['lr'], lrDecay=config['lr_decay'], weighted=config['weighted'])
trainer.model.load_state_dict(torch.load(args.model_dir))
predicts = trainer.test_network(testLoader)

with open(args.output_dir, 'w') as f:
    f.write('Id,Predicted\n')
    for key, value in predicts.items():
        f.write(f'{key},{value}\n')
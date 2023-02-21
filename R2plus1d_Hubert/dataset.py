from torch.utils.data import Dataset
import os
import numpy as np
import torch
import random
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from custom_transform import *

class TTMDataset(Dataset):
    def __init__(self, data_path, config, transform, mode="test"):
        self.transform = transform
        self.audio_length_limit = config['audio_length_limit']
        self.video_length_limit = config['video_length_limit']
        self.mode = mode
        self.data_path = data_path
        print(f"Len of {self.mode} dataset : {len(self.data_path)}")
    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, index):
        path = self.data_path[index]
        try:
            data = np.load(path)
        except:
            raise
        
        video = data['image']
        audio = data['audio']

        origin_length = len(audio)
        if origin_length > 16000 * self.audio_length_limit:
            audio_length = int(16000 * self.audio_length_limit)
            audio = audio[:audio_length]
            # audio_norm = (audio - np.mean(audio)) / np.std(audio)
            # audio = torch.FloatTensor(audio_norm) 
        else:
            audio_length = origin_length
        #     new_audio = torch.zeros(int(16000 * self.audio_length_limit))
        #     audio_norm = (audio - np.mean(audio)) / np.std(audio)
        #     new_audio[:audio_length] = torch.FloatTensor(audio_norm) 
        #     audio = new_audio       

        # video = torch.FloatTensor(video)
        if len(video) == 0:
            video = torch.zeros([self.video_length_limit, 3, 96, 96])
        elif len(video) < self.video_length_limit:
            new_video = torch.zeros([self.video_length_limit, 3, 96, 96])
            new_video[:len(video)] = torch.FloatTensor(video)
            video = new_video
            video = torch.FloatTensor(video)
        else:
            # video = video[:self.video_length_limit]
            video = video[np.round(np.linspace(0, len(video)-1, num=self.video_length_limit)).astype(int)]
            video = torch.FloatTensor(video)
        video = torch.einsum("jklm->kjlm", video)
        if self.transform is not None:
            video = self.transform(video)
        if self.mode != 'test':
            label = float(data['ttm'])
            return video, audio, audio_length, torch.tensor(label)
        else:
            return video, audio, audio_length, os.path.basename(path)[:-4]

    def collate(self, batch):
        vals = list(zip(*batch))
        collated = {}
        collated['video'] = torch.stack(vals[0]).cuda()
        collated['audio'] = vals[1]
        collated['audio_length'] = torch.LongTensor(vals[2])
        collated['audio_attention_mask'] = torch.arange(len(collated['audio'][0])).unsqueeze(0) >= collated['audio_length'].unsqueeze(1)
        if self.mode != 'test':
            collated['label'] = torch.stack(vals[3])
        else:
            collated['id'] = vals[3]
        return collated

class TTMDatasetOnlyVideo(Dataset):
    def __init__(self, data_path, config, transform, mode="test"):
        self.transform = transform
        self.video_length_limit = config['video_length_limit']
        self.mode = mode
        self.data_path = []
        for path in tqdm(data_path):
            data = np.load(path)
            if (len(data['image']) != 0):
                self.data_path.append(path)
        print(f"Len of {self.mode} dataset : {len(self.data_path)}")
    
    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, index):
        path = self.data_path[index]
        try:
            data = np.load(path)
        except:
            raise
        
        video = data['image']

        if len(video) > self.video_length_limit:
            video = video[np.round(np.linspace(0, len(video)-1, num=self.video_length_limit)).astype(int)]
        
        video = torch.FloatTensor(video)
        # print(video.shape)
        if self.transform is not None:
            transform_num = random.randint(0, len(self.transform))
            transform_set = random.sample(self.transform, transform_num)
            size = int(random.uniform(0.75, 1) * 96)
            angle = random.uniform(-5, 5)
            for i in range(len(video)):
                frame = video[i]
                for t in transform_set:
                    if t == "hflip":
                        frame = horizontal_flip(frame)
                    if t == "blur":
                        frame = gaussian_blur(frame)
                    if t == "gray":
                        frame = grayscale(frame)
                    if t == "crop":
                        frame = crop(frame, size)
                    if t == "rotate":
                        frame = rotate(frame, angle)
                video[i] = frame
        
        if self.mode != 'test':
            label = int(data['ttm'])
            return video, torch.tensor(label)
        else:
            return video, os.path.basename(path)[:-4]

    def collate(self, batch):
        vals = list(zip(*batch))
        collated = {}
        collated['video'] = pad_sequence(vals[0], batch_first=True, padding_value=0.5)
        collated['video'] = torch.einsum("bjklm->bkjlm", collated['video'])
        if self.mode != 'test':
            collated['label'] = torch.stack(vals[1])
        else:
            collated['id'] = vals[1]
        return collated

class TTMDatasetTimesformer(Dataset):
    def __init__(self, data_path, config, transform, mode="test"):
        self.transform = transform
        self.video_length_limit = config['video_length_limit']
        self.mode = mode
        self.data_path = []
        for path in tqdm(data_path):
            data = np.load(path)
            if (len(data['image']) != 0):
                self.data_path.append(path)
        print(f"Len of {self.mode} dataset : {len(self.data_path)}")
    
    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, index):
        path = self.data_path[index]
        data = np.load(path)
    
        video = data['image']

        if len(video) == 0:
            raise
        elif len(video) > self.video_length_limit:
            video = video[np.round(np.linspace(0, len(video)-1, num=self.video_length_limit)).astype(int)]
            # video = torch.FloatTensor(video)
        else:
            new_video = np.full([self.video_length_limit, 3, 96, 96], 0.5)
            new_video[:len(video)] = video
            video = new_video
            # video = torch.FloatTensor(video)
        
        if self.transform is not None:
            video = self.transform(video)
        if self.mode != 'test':
            label = int(data['ttm'])
            return list(video), torch.tensor(label)
        else:
            return list(video), os.path.basename(path)[:-4]

    def collate(self, batch):
        vals = list(zip(*batch))
        collated = {}
        collated['video'] = vals[0]
        if self.mode != 'test':
            collated['label'] = torch.stack(vals[1])
        else:
            collated['id'] = vals[1]
        return collated


class TTMDatasetWithBlank(Dataset):
    def __init__(self, root_dir, mode="train", random_seed=777):
        self.root_dir = root_dir
        self.files = [i for i in os.listdir(root_dir) if i.endswith(".npz")]
        random.seed(random_seed)
        random.shuffle(self.files)
        self.mode = mode # train / test

    def __len__(self):
        if self.mode == "train":
            return round(len(self.files) * 0.9)
        else:
            return round(len(self.files) * 0.1)

    def __getitem__(self, index):
        if self.mode != "train":
            index = index + round(len(self.files) * 0.9)
        data = np.load(os.path.join(self.root_dir, self.files[index]))
        image, audio, ttm, is_empty = data['image'], data['audio'], data['ttm'], data['is_empty']

        # process image frames
        full_image = np.zeros((len(is_empty), 3, 96, 96))
        k = 0
        for index, empty in enumerate(is_empty):
            if empty:
                full_image[index] += 0.5 # images are normalized to mean 0.5, std 0.5
            else:
                full_image[index] = image[k]
                k += 1
	
        return torch.from_numpy(full_image), torch.from_numpy(audio), torch.from_numpy(ttm)
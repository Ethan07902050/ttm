#encoding: utf-8
import os
import cv2
import glob
import random
import scipy.io
import numpy as np
import pickle as pl

import csv
import torchaudio
import torchvision.transforms as transforms
from math import floor
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
import librosa
from natsort import natsort

class ttm_dataset():
    def __init__(self, folds, data_dir, split_ratio, max_frame_num=128, max_audio_len=200000):
        self.folds = folds
        self.split_ratio = split_ratio
        self.data_dir = data_dir
        self.filenames = []
        self.max_frame_num = max_frame_num
        self.max_audio_len = max_audio_len
        self.empty_frame = torch.zeros(1, 3, 96, 96)
        self.empty_audio = np.zeros((2, 1000))

        for input_dir in data_dir:
            print(input_dir)
            all_files = os.listdir(input_dir)
            total = len(all_files)
            sep = int(total * split_ratio)
            print(total, sep)
            if folds == 'train':
                all_files = all_files[:sep]
            elif folds == 'val':
                all_files = all_files[sep:]

            for filename in tqdm(all_files):
                self.filenames.append(os.path.join(input_dir, filename))
            
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        data = np.load(filename, allow_pickle=True)

        try:
            clip_size = data['clip_frame'].shape
            if int(clip_size[0]) == 0:
                clip = self.empty_frame
            else:
                clip = torch.from_numpy(data['clip_frame'][:self.max_frame_num, :, :, :])
        except EOFError:
            print("Video exception!")
            clip = self.empty_frame

        try:
            audio = data['audio'][:, :self.max_audio_len]
            if audio.shape[1] < 1000:
                audio = self.empty_audio
        except:
            print("Audio exception!")
            audio = self.empty_audio


        label = torch.from_numpy(data['label'])
        # except:
        #     label = torch.tensor(0.)

        return clip, audio, label

    def __len__(self):
        return len(self.filenames)

class ttm_test():
    def __init__(self, folds, data_dir, seg_dir, split_ratio, max_frame_num=128, max_audio_len=200000):
        self.folds = folds
        self.split_ratio = split_ratio
        self.data_dir = data_dir
        self.filenames = []
        self.mapping = {}
        self.max_frame_num = max_frame_num
        self.max_audio_len = max_audio_len
        self.empty_frame = torch.zeros(1, 3, 96, 96)
        self.empty_audio = np.zeros((2, 1000))
        self.clip_prefix = [filename.split("_")[0] for filename in os.listdir(seg_dir)]

        # the npz files
        input_dir = data_dir
        print(input_dir)
        all_files = os.listdir(input_dir)
        all_files = natsort.natsorted(all_files)
        total = len(all_files)

        for filename in tqdm(all_files):
            self.filenames.append(os.path.join(input_dir, filename))

        for i, prefix in tqdm(enumerate(self.clip_prefix)):
            seg_path = os.path.join(seg_dir, prefix + "_seg.csv")

            with open(seg_path, newline='') as csvfile:
                rows = csv.reader(csvfile)
                next(rows, None)  # skip the headers
                count = 0
                for row in rows:
                    npz_path = os.path.join(self.data_dir, prefix + f"_{count}.npz")

                    person_id, start_frame, end_frame = row

                    submit_id = f"{prefix}_{person_id}_{start_frame}_{end_frame}" 
                    self.mapping[prefix + f"_{count}.npz"] = submit_id

                    count = count + 1

            
            
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        submit_id = self.mapping[filename.split("/")[-1]]
        data = np.load(filename, allow_pickle=True)


        try:
            clip_size = data['clip_frame'].shape
            if int(clip_size[0]) == 0:
                clip = self.empty_frame
            else:
                clip = torch.from_numpy(data['clip_frame'][:self.max_frame_num, :, :, :])
        except EOFError:
            print("Video exception!")
            clip = self.empty_frame

        try:
            audio = data['audio'][:, :self.max_audio_len]
            if audio.shape[1] < 1000:
                audio = self.empty_audio
        except:
            print("Audio exception!")
            audio = self.empty_audio

        # except:
        #     label = torch.tensor(0.)

        return clip, audio, submit_id

    def __len__(self):
        return len(self.filenames)
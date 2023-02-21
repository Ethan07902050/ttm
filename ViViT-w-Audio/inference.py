import os, sys
import torch
from torch.utils.data import Dataset
import torchaudio
import numpy as np
from tqdm import tqdm
from ViViT_w_Audio import ViViT_w_Audio_v1
import csv

###
test_data_dir = sys.argv[1]
out_csv = sys.argv[2]
best_model_dir = "models_file/best_vivit_w_audio.pth"
###

class TTM_Inference_Dataset(Dataset):
    def __init__(self, root_dir, n_mfcc=13):
        self.root_dir = root_dir
        self.files = [i for i in os.listdir(root_dir) if i.endswith(".npz")]
        self.n_mfcc = n_mfcc

    def __len__(self):
        return len(self.files)

    def transform_audio(self, audio):
        hop_length = 532 # to match the time of each frame as possible
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate = 16000,
            n_mfcc = self.n_mfcc,
            melkwargs={
                "n_fft": 2048,
                "n_mels": 256,
                "hop_length": hop_length,
                "mel_scale": "htk",
            },
        )
        return mfcc_transform(audio).T # transpose: time first

    def __getitem__(self, index):

        ID = self.files[index].split(".")[0]

        data = np.load(os.path.join(self.root_dir, self.files[index]))
        image, audio, is_empty = data['image'], data['audio'], data['is_empty']

        # process image frames
        full_image = np.zeros((len(is_empty), 3, 96, 96))
        k = 0
        for index, empty in enumerate(is_empty):
            if empty:
                full_image[index] += 0.5 # images are normalized to mean 0.5, std 0.5
            else:
                full_image[index] = image[k]
                k += 1

        # transform audio waveform to mfcc
        if len(is_empty) > 2:
            mfcc = self.transform_audio(torch.from_numpy(audio))[:len(is_empty),:]
        else:
            mfcc = torch.zeros((len(is_empty), self.n_mfcc))

        return ID, torch.from_numpy(full_image), mfcc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(best_model_dir, map_location=device)
print(f"Checkpoint loaded from: {best_model_dir}")
saved_cfg = ckpt['cfg']
print(f"config: {saved_cfg}")

model = ViViT_w_Audio_v1(
    image_size_H=saved_cfg['image_size_H'],
    image_size_W=saved_cfg['image_size_W'],
    patch_size_h=saved_cfg['patch_size_h'],
    patch_size_w=saved_cfg['patch_size_w'],
    audio_dim=saved_cfg['audio_dim'],
    max_num_frames=saved_cfg['max_num_frames'],
    dim=saved_cfg['dim'],
    depth=saved_cfg['depth'],
    heads=saved_cfg['heads'],
    pool=saved_cfg['pool'],
    in_channels=saved_cfg['in_channels'],
    dim_head=saved_cfg['dim_head'],
    dropout=saved_cfg['dropout'],
    emb_dropout=saved_cfg['emb_dropout'],
    scale_dim=saved_cfg['scale_dim'],
    audio_scale=saved_cfg['audio_scale']
).to(device)
model.load_state_dict(ckpt['model'])
model.eval()

dataset = TTM_Inference_Dataset(root_dir=test_data_dir, n_mfcc=saved_cfg['audio_dim'])
print(f"Writing into {out_csv}")
with open(out_csv, "w") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["Id", "Predicted"])

    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            ID, image, audio = dataset[i]
            image, audio = image.unsqueeze(0).to(device).to(torch.float32), audio.unsqueeze(0).to(device)

            # seq len may be larger than max_num_frames
            cut = (audio.shape[1] - 1) // (saved_cfg['max_num_frames'])

            # seg_num = cut + 1
            # frame_per_seg = int(audio.shape[1] / (cut + 1))
            cur_seg_frame_num = saved_cfg['max_num_frames']
            start = 0
            score = 0
            while(cut > 0):
                score = model(
                    image[:, start:(start + cur_seg_frame_num), :,:,:], 
                    audio[:, start:(start + cur_seg_frame_num), :]
                ) * cur_seg_frame_num
                start = start + cur_seg_frame_num
                cut -= 1

            score = model(image[:, start:, :,:,:], audio[:, start:, :]) * (audio.shape[1] - start + 1)
            pred = round((score / audio.shape[1]).item()) # weighted score

            writer.writerow([
                ID, pred
            ])

            

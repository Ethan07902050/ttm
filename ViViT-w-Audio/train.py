import torch
import torch.nn as nn
from tqdm import tqdm
import os, sys, random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchvision import transforms
import librosa
from ViViT_w_Audio import ViViT_w_Audio_v1 # ViViT_w_Audio_v2
from config import config
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###
train_data_dir = sys.argv[1]

if not os.path.isdir("models_file"):
    os.makedirs("models_file")

best_model_dir = "models_file/best_vivit_w_audio.pth"
model_dir = "models_file/ckpt_vivit_w_audio.pth"
# data_dir = "/home/kszuyen/DLCV/final/data"
###

class dataset(Dataset):
    def __init__(self, root_dir, n_mfcc=13, mode="train", random_seed=999, max_num_frames=1200):
        self.root_dir = root_dir
        self.files = [i for i in os.listdir(root_dir) if i.endswith(".npz")]
        random.seed(random_seed)
        random.shuffle(self.files)
        self.mode = mode # train / test
        self.n_mfcc = n_mfcc
        self.max_num_frames = max_num_frames

    def __len__(self):
        if self.mode == "train":
            return round(len(self.files) * 0.9)
        else:
            return round(len(self.files) * 0.1)

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

    def audio_augmentation(self, audio):
        """  data augmentation for audio waveform  """
        input_length = len(audio)

        ns = np.random.choice([0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006]) # add noise
        if ns:
            wn = np.random.randn(input_length)
            audio = audio + ns*wn

        if np.random.choice([0, 1], p=[0.7, 0.3]):
            roll = np.random.choice(list(range(-30, 30)))
            audio = np.roll(audio, roll)

        if np.random.choice([0, 1], p=(0.2, 0.8)) and input_length > 2048:
            rate = np.random.choice([0.7, 0.8, 0.9, 1.1, 1.2, 1.3])
            audio = librosa.effects.time_stretch(audio, rate=rate)

            if len(audio)>input_length:
                audio = audio[:input_length]
            else:
                audio = np.pad(audio, (0, max(0, input_length - len(audio))), "constant")
        return audio

    def __getitem__(self, index):
        if self.mode != "train":
            index = index + round(len(self.files) * 0.9)
        data = np.load(os.path.join(self.root_dir, self.files[index]))
        image, audio, ttm, is_empty = data['image'], data['audio'], data['ttm'], data['is_empty']

        # image data augmentation
        if self.mode == "train" and len(image) != 0:
            # random horizontal flip
            if random.choice([0, 1]):
                np.fliplr(image)
            image = (image - np.min(image))/(np.max(image) - np.min(image))
            angle = random.uniform(-20, 20)
            distort_ori = random.choice(['ver', 'hor'])
            distort_x = random.uniform(0, 0.04)
            distort_y = random.choice(range(9))
            ratio_r = random.uniform(0.8, 1)
            ratio_g = random.uniform(0.8, 1)
            ratio_b = random.uniform(0.8, 1)

            for i in range(image.shape[0]):
                temp_image = np.transpose(image[i], (1, 2, 0))

                temp_image = utils.distort(temp_image, orientation=distort_ori, x_scale=distort_x, y_scale=distort_y)
                temp_image = utils.rotate_img(temp_image, angle)
                temp_image = utils.change_channel_ratio(temp_image, channel='r', ratio=ratio_r)
                temp_image = utils.change_channel_ratio(temp_image, channel='g', ratio=ratio_g)
                temp_image = utils.change_channel_ratio(temp_image, channel='b', ratio=ratio_b)
                image[i] = np.transpose(temp_image, (2, 0, 1))

        img_transform = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])

        # process image frames
        full_image = torch.zeros((len(is_empty), 3, 96, 96))
        k = 0
        for index, empty in enumerate(is_empty):
            if empty:
                full_image[index] += 0.5 # images are normalized to mean 0.5, std 0.5
            else:
                full_image[index] = img_transform(torch.from_numpy(image[k]))
                k += 1

        # transform audio waveform to mfcc
        if self.mode == "train":
            audio = self.audio_augmentation(audio).astype('float32')
        if len(is_empty) > 2:
            mfcc = self.transform_audio(torch.from_numpy(audio))[:len(is_empty),:]
        else:
            mfcc = torch.zeros((len(is_empty), self.n_mfcc))

        if len(is_empty) > self.max_num_frames:
            start = random.choice(range(len(is_empty) - self.max_num_frames))
            return full_image[start:start+self.max_num_frames], mfcc[start:start+self.max_num_frames], torch.from_numpy(ttm)

        return full_image, mfcc, torch.from_numpy(ttm)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    pbar = tqdm(loader)
    for images, audios, targets in pbar:
        images, audios, targets = images.to(device).to(torch.float32), audios.to(device), targets.to(device).to(torch.float32)
        mini_batch_size = targets.shape[0]
        scores = model(images, audios)

        for i in range(mini_batch_size):  
            if np.random.choice([0, 1], p=[0.95, 0.05]): # random flip label
                targets[i] = 1 if targets[i]==0 else 0
            if targets[i] == 0: # label smoothing
                targets[i] += ((0.1) * torch.rand(1)).item()
            else:
                targets[i] -= ((0.25) * torch.rand(1)).item()

        targets = targets.unsqueeze(1)
        
        loss = criterion(scores, targets) # match shape [batch_size, 1]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix({"loss": loss.item()})
def evaluate(model, loader):
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, audios, targets in tqdm(loader):
            mini_batch_size = targets.shape[0]
            images, audios = images.to(device), audios.to(device)

            scores = model(images.to(torch.float32), audios)
            pred = np.round(scores.detach().cpu().numpy())
            correct += (pred == targets.unsqueeze(1).numpy()).sum()
            total += mini_batch_size
        accuracy = correct / total
        print(f"Accuracy: {accuracy:.3f} ({correct}/{total})")
    return accuracy

def save_model(model, optimizer, epoch, best_acc, cfg, model_dir):
    cfg_dict = dict(
        image_size_H=cfg.image_H,
        image_size_W=cfg.image_W,
        patch_size_h=cfg.patch_size_h,
        patch_size_w=cfg.patch_size_w,
        audio_dim=cfg.audio_dim,
        max_num_frames=cfg.max_num_frames,
        dim=cfg.dim,
        depth=cfg.depth,
        heads=cfg.heads,
        pool=cfg.pool,
        in_channels=cfg.in_channels,
        dim_head=cfg.dim_head,
        dropout=cfg.dropout,
        emb_dropout=cfg.emb_dropout,
        scale_dim=cfg.scale_dim,
        audio_scale=cfg.audio_scale
    )
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "acc": best_acc,
        "cfg": cfg_dict
    }, model_dir)

cfg = utils.dotdict(config)

train_dataset = dataset(
    train_data_dir, 
    mode="train", 
    n_mfcc=cfg.audio_dim,
    max_num_frames=cfg.max_num_frames
)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    collate_fn=utils.MyCollate(),
    num_workers=cfg.num_workers
)

val_dataset = dataset(
    cfg.data_dir, 
    mode="val", 
    n_mfcc=cfg.audio_dim, 
    max_num_frames=cfg.max_num_frames
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
    collate_fn=utils.MyCollate(),
    num_workers=cfg.num_workers
)

model = ViViT_w_Audio_v1(
    image_size_H=cfg.image_H,
    image_size_W=cfg.image_W,
    patch_size_h=cfg.patch_size_h,
    patch_size_w=cfg.patch_size_w,
    audio_dim=cfg.audio_dim,
    max_num_frames=cfg.max_num_frames,
    dim=cfg.dim,
    depth=cfg.depth,
    heads=cfg.heads,
    pool=cfg.pool,
    in_channels=cfg.in_channels,
    dim_head=cfg.dim_head,
    dropout=cfg.dropout,
    emb_dropout=cfg.emb_dropout,
    scale_dim=cfg.scale_dim,
    audio_scale=cfg.audio_scale
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
# optimizer = torch.optim.Adam(
#     [
#     # {"params": model.audio_net.parameters(), "lr": 5e-5},
#     {"params": model.space_transformer.parameters(), "lr": 2e-4},
#     # {"params": model.audio_transformer.parameters(), "lr": 2e-4},
#     {"params": model.temporal_transformer.parameters(), "lr": 3e-4},
#     # {"params": model.mlp_head.parameters(), "lr": 1e-4},
#     ],
#     lr=3e-4
#     )
criterion = nn.BCELoss() # nn.MSELoss()

if cfg.load_saved_model and os.path.isfile(cfg.model_dir):
    print("Loading checkpoint...")
    ckpt = torch.load(cfg.model_dir, map_location=device)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    start_epoch = ckpt['epoch']
    best_acc = ckpt['acc']
else:
    best_acc = 0
    start_epoch = 0

print("Start training:")
for epoch in range(start_epoch+1, cfg.num_epoch+1):
    print(f"Epoch {epoch}:")
    train_one_epoch(model, train_loader, optimizer, criterion)
    save_model(model, optimizer, epoch, best_acc, cfg, cfg.model_dir)
    print("Checkpoint saved!")

    print("Evaluate...")
    accuracy = evaluate(model, val_loader)
    if accuracy > best_acc:
        best_acc = accuracy
        save_model(model, optimizer, epoch, best_acc, cfg, best_model_dir)
        print("Best model saved!")





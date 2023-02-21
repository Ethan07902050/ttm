import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from module import Attention, PreNorm, FeedForward
import numpy as np
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
import torch.optim as optim
from transformers import Wav2Vec2Processor
import torch.nn.utils.rnn as rnn_utils
import librosa
from pathlib import Path
import random

from model import TTM
from dataset import ttm_dataset, ttm_test
from get_logger import get_logger



processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", 
            cache_dir="")

def ttm_collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    clips, audios, labels = [], [], []

    # Gather in lists, and encode labels as indices
    for clip, audio, label in batch:
        clips += [clip]
        # print(type(audio))
        # print(audio)
        audios.append(librosa.to_mono(audio))
        labels += [label]

    # Group the list of tensors into a batched tensor
    clips = rnn_utils.pad_sequence(clips, batch_first=True)
    # print(audios)
    audios = processor(audios, sampling_rate=16000, padding=True, return_tensors="pt")
    labels = torch.stack(labels)

    return clips, audios, labels

def ttm_test_collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    clips, audios, submit_ids = [], [], []

    # Gather in lists, and encode labels as indices
    for clip, audio, submit_id  in batch:
        clips += [clip]
        audios.append(librosa.to_mono(audio))
        submit_ids += [submit_id]

    # Group the list of tensors into a batched tensor
    clips = rnn_utils.pad_sequence(clips, batch_first=True)
    # print(audios)
    audios = processor(audios, sampling_rate=16000, padding=True, return_tensors="pt")

    return clips, audios, submit_ids

# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def freeze_pretrained(model):
    for name, para in model.audio_stream.named_parameters():
        if name != "mlp_head":
            para.requires_grad_(False)
        else:
            para.requires_grad_(True)
    for name, para in model.video_stream.model.model.blocks[:-1].named_parameters():
        para.requires_grad_(False)
            
def main(args):

    # =======
    # initialize data
    # =======

    batch_size = args.batch_size
    num_epoch = args.epoch
    data_root = args.input_dir
    test_seg_path = args.input_csv
    best_model_path = "checkpoints/best_ttm_model.ckpt"
    model_path = "checkpoints/ttm_model.ckpt"
    optimizer_path = "checkpoints/optimizer.ckpt"
    log_path = "results/training.log"
    accum_iter = 32
    max_frame_num = 128
    logger = get_logger(log_path)
    same_seeds(0)
    DEVICE = get_device()

    test_set = ttm_test("train", data_root, test_seg_path, 1, max_frame_num=max_frame_num)

    test_loader = DataLoader(test_set, 
                                batch_size=batch_size,
                                num_workers=0, collate_fn=ttm_test_collate_fn,
                                shuffle=False)

    audio_replacement = processor(np.zeros((2, 1000)), sampling_rate=16000, padding=True, return_tensors="pt")
    model = TTM(device=DEVICE, audio_replacement=audio_replacement).to(DEVICE)

    model.load_state_dict(torch.load(args.checkpoint))

    # ==========================================
    # 5. Training with validation
    # ==========================================

    model.eval()

    val_accs = []
    submit_ids = []
    predictions = []

    for batch in tqdm(test_loader):
        clips, audios, ids = batch
        with torch.no_grad():
            logits = model(clips.to(DEVICE), audios.to(DEVICE))

        # acc = (logits.argmax(dim=-1) == labels.to(DEVICE)).float().mean()
        for pred in logits.argmax(dim=-1):
            predictions.append(int(pred))
        submit_ids.append(ids)
            

        # val_accs.append(acc)

    with open(args.output_path, "w") as f:

        # The first row must be "Id, Category"
        f.write("Id,Predicted\n")
    
        for i, pred in  enumerate(predictions):
            f.write(f"{submit_ids[i][0]},{pred}\n")
    # val_acc = sum(val_accs) / len(val_accs)

    # logger.info(f"[ Valid ]  acc = {val_acc:.5f}")

    
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=Path,
        help="data dir path",
        required=True,
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        help="input csv path",
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batchsize",
        default=1,
        required=False,
    )
    parser.add_argument(
        "--epoch",
        type=int,
        help="Number of epoch",
        default=300,
        required=False,
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Pred path",
        default="./pred.csv",
        required=False,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Pred path",
        required=True,
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
    
    
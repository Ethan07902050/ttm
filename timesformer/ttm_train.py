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

from model import TTM
from dataset import ttm_dataset
from get_logger import get_logger



processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft", 
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

    # for name, para in model.audio_stream.extractor.encoder.layers[12:].named_parameters():
    #     para.requires_grad_(True)

    for name, para in model.video_stream.model.model.blocks[:-1].named_parameters():
        para.requires_grad_(False)
            
def main(args):

    # =======
    # initialize data
    # =======

    batch_size = args.batch_size
    num_epoch = args.epoch
    data_root = [args.input_dir]
    best_model_path = f"checkpoints/best_ttm_model_{args.ver}.ckpt"
    model_path = f"checkpoints/ttm_model_{args.ver}.ckpt"
    optimizer_path = f"checkpoints/optimizer_{args.ver}.ckpt"
    log_path = f"results/training_{args.ver}.log"
    accum_iter = 64
    max_frame_num = 128
    logger = get_logger(log_path)
    same_seeds(0)
    DEVICE = get_device()

    train_set = ttm_dataset("train", data_root, 0.9, max_frame_num=max_frame_num)
    val_set = ttm_dataset("val", data_root, 0.9, max_frame_num=max_frame_num)

    train_loader = DataLoader(train_set, 
                                batch_size=batch_size,
                                num_workers=0, collate_fn=ttm_collate_fn,
                                shuffle=True)
    val_loader = DataLoader(val_set, 
                                batch_size=batch_size,
                                num_workers=0, collate_fn=ttm_collate_fn,
                                shuffle=False)

    audio_replacement = processor(np.zeros((2, 1000)), sampling_rate=16000, padding=True, return_tensors="pt")
    model = TTM(device=DEVICE, audio_replacement=audio_replacement).to(DEVICE)

    # print(model)
    
    # parameters to be updated 
    # freeze_pretrained(model)
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of params: {n_parameters}")

    param_dicts = [
        {"params": [p for n, p in model.named_parameters(
        ) if p.requires_grad]},
    ]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(param_dicts, lr=0.005, weight_decay=0.0001)

    # ==========================================
    # 5. Training with validation
    # ==========================================

    best_acc = 0.0

    for epoch in range(num_epoch):

        model.train()

        train_loss = []
        train_accs = []

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            torch.cuda.empty_cache()

            clips, audios, labels = batch
            logits = model(clips.to(DEVICE), audios.to(DEVICE))
            # print(logits.shape)
            # print("Logits: ", logits, labels)
            loss = criterion(logits, labels.to(DEVICE))
        
            loss = loss / accum_iter 
            loss.backward()

            acc = (logits.argmax(dim=-1) == labels.to(DEVICE)).float().mean()
            # print("Results: ", logits.round(), labels)
        

            train_loss.append(loss.item())
            train_accs.append(acc)

            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
        
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        logger.info(f"[ Train | {epoch + 1:03d}/{num_epoch:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        model.eval()

        val_loss = []
        val_accs = []

        for batch in tqdm(val_loader):
            clips, audios, labels = batch
            with torch.no_grad():
                logits = model(clips.to(DEVICE), audios.to(DEVICE))
            loss = criterion(logits, labels.to(DEVICE))

            acc = (logits.argmax(dim=-1) == labels.to(DEVICE)).float().mean()

            val_loss.append(loss.item())
            val_accs.append(acc)
        
        val_loss = sum(val_loss) / len(val_loss)
        val_acc = sum(val_accs) / len(val_accs)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            logger.info("Saving model with acc {:.5f}".format(val_acc))
        
        torch.save(model.state_dict(), model_path)
        torch.save(optimizer.state_dict(), optimizer_path)
    

        logger.info(f"[ Valid | {epoch + 1:03d}/{num_epoch:03d} ] loss = {val_loss:.5f}, acc = {val_acc:.5f}")
        logger.info('Best acc so far {:.5f}'.format(best_acc))

    
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=Path,
        help="input data path",
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
        "--ver",
        type=str,
        help="description of model",
        default="",
        required=False,
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
    
    
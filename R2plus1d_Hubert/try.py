from decord import VideoReader, cpu
import torch
import numpy as np

from transformers import VideoMAEFeatureExtractor, TimesformerForVideoClassification
from huggingface_hub import hf_hub_download

np.random.seed(0)


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


# video clip consists of 300 frames (10 seconds at 30 FPS)
file_path = hf_hub_download(
    repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
)
videoreader = VideoReader(file_path, num_threads=1, ctx=cpu(0))

# sample 8 frames
videoreader.seek(0)
indices = sample_frame_indices(clip_len=8, frame_sample_rate=4, seg_len=len(videoreader))
video = videoreader.get_batch(indices).asnumpy()

feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")
print(video.shape)
inputs = feature_extractor(list(video), return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# model predicts one of the 400 Kinetics-400 classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
# ViViT-w-Audio

* code modified and based on ViViT: https://github.com/rishikksh20/ViViT-pytorch/blob/master/vivit.py
* early video and audio fusion

## Preprocess

See repo root README

## How to run ViViT-w-Audio
* modify `config.py` to set parameters
* you can change model from ViViT_w_Audio_v1 to ViViT_w_Audio_v2, may have different results

### Train
```bash
$ bash train.sh <Path to npz train data directory>
```

### Inference
```bash
$ bash inference.sh <Path to npz test data directory> <Path to prediction csv>
```

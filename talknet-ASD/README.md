# TalkNet-ASD

* Middle fusion
* Serparete encoder (temporal encoder)
* Use cross attention & self attention to fuse different modalities' information
* reference code : https://github.com/TaoRuijie/TalkNet-ASD

## Preprocess

See repo root README

## How to run TalkNet-ASD?

### Train

```bash
$bash train.sh <Path to npz train data directory> <Path to directory for saving model weights>
```

### Inference

```bash
$bash inference.sh <Path to npz test data directory> <Path to prediction csv> <Path to model weight to load>
```

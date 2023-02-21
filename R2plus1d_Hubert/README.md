# R(2+1)d + Hubert

* Late fusion
* Serparete encoder (R(2+1)d & Hubert)

## Preprocess

See repo root README

## How to run R(2+1)d + Hubert?

### Train

```bash
$bash train.sh <Path to npz train data directory> <Path to directory for saving model weights>
```

### Test

```bash
$bash inference.sh <Path to npz test data directory> <Path to prediction csv> <Path to model weight to load>
```

## How to run only R(2+1)d (Single modal)?

### Train

```bash
$bash train_only_video.sh <Path to npz train data directory> <Path to directory for saving model weights>
```

### Inference

```bash
$bash inference_only_video.sh <Path to npz test data directory> <Path to prediction csv> <Path to model weight to load>
```

# TimeSformer + wav2vec

* code modified and based on TimeSformer: https://github.com/facebookresearch/TimeSformer
* Middle audio and video fusion 

## Preprocess

See repo root README

## How to run ViViT-w-Audio


### Train
* set the parameters in ttm_train.py
```bash
$ bash train.sh <Path to npz train data directory>
```

### Inference
* set the parameters in ttm_test.py
```bash
$ bash test.sh <Path to npz test data directory> <Path to prediction csv>
```

# AV-Hubert

## Preprocess
See repo root README

## Installation
```bash
$git submodule update --init --recursive
$cd av_hubert
$pip install -r requirements.txt
$cd fairseq
$pip install --editable ./
```

## How to run AV-Hubert?
Change directory to `av_hubert/avhubert`.

### Download
```bash
$bash download.sh
```

This will create a directory (`dlcv-final`), which contains model weights, under curreny directory.  

### Train

```bash
$bash train.sh <Path to npz train data directory> <Path to directory for saving results>
```

### Inference
```bash
$ bash inference.sh <Path to npz test data directory> <Path to prediction csv> 
```

Example :

```bash
$ bash inference.sh /tmp/data/dlcvchallenge1_test_data ./pred.csv
```


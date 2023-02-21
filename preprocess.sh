#!/bin/bash

# make audio folder
python3 preprocess_video2wav.py $1 $6

# make training data to npz format
python3 preprocess_train2npz.py $1 $2 $3 $6

# make testing data to npz format 
python preprocess_test2npz.py $1 $4 $5 $6
#!/bin/bash

# make audio folder
python3 preprocess_video2wav.py $1 $4

# make testing data to npz format 
python preprocess_test2npz.py $1 $2 $3 $4
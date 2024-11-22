#!/bin/bash

dataset="$1"

for method in ganomaly draem fastflow padim patchcore reverse stfpm
do
    python src/comparison/regional/evaluate.py $method --test_data $dataset --model_dir models/$dataset/classify/ --decoder_dir models/$dataset/inpaint/ --data_file results/$dataset/regional/data/$method.csv --estimator_file models/$dataset/anomaly/$method.p
done

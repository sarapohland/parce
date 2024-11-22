#!/bin/bash

dataset="$1"

for method in parce softmax temperature dropout ensemble energy odin openmax dice kl mahalanobis knn
do
    python src/comparison/overall/evaluate.py $method --test_data $dataset --model_dir models/$dataset/classify/ --decoder_dir models/$dataset/reconstruct/ --data_file results/$dataset/unmodified/data/$method.csv --estimator_file models/$dataset/overall/$method.p
done

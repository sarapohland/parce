#!/bin/bash

dataset="$1"

for factor in 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0 20.0
do
    python src/analysis/evaluate.py --test_data $dataset --model_dir models/$dataset/classify/ --decoder_dir models/$dataset/reconstruct/ --property saturation --factor $factor --data_dir results/$dataset/modified/data/
done

for factor in 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0 3.25 3.5 3.75 4.0 4.25 4.5 4.75 5.0
do
    python src/analysis/evaluate.py --test_data $dataset --model_dir models/$dataset/classify/ --decoder_dir models/$dataset/reconstruct/ --property contrast --factor $factor --data_dir results/$dataset/modified/data/
done

for factor in 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0 3.25 3.5 3.75 4.0 4.25 4.5 4.75 5.0
do
    python src/analysis/evaluate.py --test_data $dataset --model_dir models/$dataset/classify/ --decoder_dir models/$dataset/reconstruct/ --property brightness --factor $factor --data_dir results/$dataset/modified/data/
done

for factor in 0 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200
do
    python src/analysis/evaluate.py --test_data $dataset --model_dir models/$dataset/classify/ --decoder_dir models/$dataset/reconstruct/ --property noise --factor $factor --data_dir results/$dataset/modified/data/
done

for factor in 0.5 2.0 4.0 6.0 8.0 10.0 12.0 14.0 16.0 18.0 20.0 22.0 24.0 26.0 28.0 30.0 32.0 34.0 36.0 38.0 40.0
do
    python src/analysis/evaluate.py --test_data $dataset --model_dir models/$dataset/classify/ --decoder_dir models/$dataset/reconstruct/ --property pixelate --factor $factor --data_dir results/$dataset/modified/data/
done
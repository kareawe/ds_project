#!/usr/bin/env bash
set -euo pipefail

python source/build_dataset.py data/*.mat --output-dir data/canonical

python source/select_feature.py \
  --canonical-dir data/canonical \
  --config source/feature_config.compromise11.json \
  --output-dir data/features/compromise11

python source/split.py \
  --dataset-csv data/features/compromise11/dataset.csv \
  --train-batch 2017-05-12_batchdata_updated_struct_errorcorrect \
  --external-test-batch 2018-02-20_batchdata_updated_struct_errorcorrect \
  --output-dir data/experiments/compromise11_2017_holdout_cv

python source/train_elastic_net.py \
  --split-dir data/experiments/compromise11_2017_holdout_cv \
  --output-dir data/experiments/compromise11_2017_holdout_cv/elastic_net_log_upper_clip_tuned_ape \
  --selection-metric avg_percent_error \
  --lambda-grid 0.001 0.002 0.005 0.01 0.02 0.05 0.1 \
  --l1-ratio-grid 0.001 0.005 0.01 0.05 0.1 0.3 0.5 \
  --zscore-clip 5.0 \
  --log-prediction-clip-mode upper

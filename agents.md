# ds-miniproject

배터리 수명 예측 실험용 최소 구조입니다. 원본 `.mat` 파일에서 시작해 canonical dataset, feature dataset, experiment split, elastic net 학습까지 재현할 수 있습니다.

원본 `.mat` 파일은 용량이 매우 커서 GitHub 저장소에는 포함하지 않습니다. 실행 전에 사용자가 직접 `data/` 아래에 배치해야 합니다.

## Structure

- `data/`: 원본 `.mat` 파일
- `preprocessing/`: 데이터 구축, split, 학습 스크립트
- `analysis/`: 실험 결과 확인용 notebook
- `eda/`: 데이터 설명 문서

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Build Canonical Dataset

```bash
python preprocessing/build_canonical_dataset.py data/*.mat --output-dir data/canonical
```

## 2. Build Feature Dataset

기본 피처셋:

```bash
python preprocessing/build_feature_dataset.py \
  --canonical-dir data/canonical \
  --output-dir data/features/default
```

축소 피처셋:

```bash
python preprocessing/build_feature_dataset.py \
  --canonical-dir data/canonical \
  --config preprocessing/feature_config.reduced_summary.json \
  --output-dir data/features/reduced_summary
```

## 3. Build Experiment Split

예시:
- train/holdout/CV: `2017-05-12`
- external test: `2018-02-20`

```bash
python preprocessing/build_experiment_split.py \
  --dataset-csv data/features/reduced_summary/dataset.csv \
  --train-batch 2017-05-12_batchdata_updated_struct_errorcorrect \
  --external-test-batch 2018-02-20_batchdata_updated_struct_errorcorrect \
  --output-dir data/experiments/reduced_summary_2017_holdout_cv
```

## 4. Train Elastic Net

```bash
python preprocessing/train_elastic_net.py \
  --split-dir data/experiments/reduced_summary_2017_holdout_cv
```

결과는 아래에 저장됩니다.

- `data/experiments/.../elastic_net_log/metrics.json`
- `analysis/experiment_registry.csv`

## Notes

- 현재 회귀 타깃은 `cycle_life`이며, 학습은 `log(cycle_life)`에서 수행한 뒤 원래 cycle scale로 평가합니다.
- 평가지표는 논문 기준으로 `RMSE (cycles)`와 `average percentage error (%)`를 사용합니다.
- 기본 실험 구조는 `train batch -> 80/20 holdout -> 4-fold CV`, 그리고 별도 `external_test_batch` 지원입니다.

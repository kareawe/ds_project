# ds_project

ESS 배터리 데이터로 cycle life를 예측하는 프로젝트입니다. 현재 저장소 루트가 바로 재현 실행 기준이며, 최종 제출 파이프라인 코드는 `source/` 아래에 정리되어 있습니다.

## 프로젝트 구조

```text
.
├── EDA/
│   └── 탐색 분석 노트북
├── analysis/
│   └── experiment_dashboard.ipynb
├── data/
│   └── README.md
├── source/
│   ├── build_dataset.py
│   ├── common.py
│   ├── feature_config.compromise11.json
│   ├── logger.py
│   ├── select_feature.py
│   ├── split.py
│   └── train_elastic_net.py
├── run_compromise11.sh
├── requirements.txt
└── Severson_NatureEnergy_2019.pdf
```

## 환경 설정

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt`에는 재현 코드 실행에 필요한 패키지와 노트북 확인용 패키지가 함께 들어 있습니다.

## 데이터 준비

원본 `.mat` 파일을 `data/` 아래에 넣어야 합니다.

예시:

```text
data/
├── 2017-05-12_batchdata_updated_struct_errorcorrect.mat
└── 2018-02-20_batchdata_updated_struct_errorcorrect.mat
```


## 재현 실행

루트에서 아래 한 줄로 전체 파이프라인을 실행할 수 있습니다.

```bash
bash run_compromise11.sh
```

이 스크립트는 아래 4단계를 순서대로 수행합니다.

1. `.mat` -> canonical dataset 변환
2. canonical dataset -> feature dataset 생성
3. 2017 batch train/holdout/CV + 2018 batch external test split 생성
4. Elastic Net 학습 및 평가

수동 실행이 필요하면 아래 명령을 그대로 사용하면 됩니다.

```bash
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
```

## 출력 산출물

- `data/canonical/`
- `data/features/compromise11/`
- `data/experiments/compromise11_2017_holdout_cv/`
- `analysis/experiment_registry.csv`
- `analysis/experiment_registry.jsonl`

## 모델링 요약

- 최종 feature config: `source/feature_config.compromise11.json`
- 최종 모델: `Elastic Net`
- target: `cycle_life`
- 학습 시 target transform: `log(cycle_life)`

EDA와 실험 결과 확인용 노트북:

- `EDA/`
- `analysis/experiment_dashboard.ipynb`

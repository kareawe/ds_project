# Submission: Compromise11

최종 제출용 최소 코드입니다.

포함 내용:
- `.mat` -> canonical dataset
- canonical -> feature dataset
- 2017 train/holdout/CV + 2018 external split
- elastic net 학습
- 최종 feature config: `source/feature_config.compromise11.json`

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

원본 `.mat` 파일을 `data/` 아래에 둔 뒤:

```bash
bash run_compromise11.sh
```

참고:
- 코드 폴더는 `preprocessing/`가 아니라 `source/`
- 파일명도 제출용으로 단순화됨
  - `build_dataset.py`
  - `select_feature.py`
  - `split.py`
  - `train_elastic_net.py`

## Outputs

- `data/canonical/`
- `data/features/compromise11/`
- `data/experiments/compromise11_2017_holdout_cv/`
- `analysis/experiment_registry.csv`

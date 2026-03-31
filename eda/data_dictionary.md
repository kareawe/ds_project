# Data Dictionary

## 목적

`.mat` 파일에서 바로 모델 입력을 만들기 전에, 어떤 데이터가 존재하는지와 어떤 계층으로 정리되는지를 문서화한다.

현재 파이프라인은 두 단계로 나뉜다.

1. `build_canonical_dataset.py`
`.mat`를 셀 단위 canonical dataset으로 변환

2. `build_feature_dataset.py`
canonical dataset에서 선택한 피처만 추출해 학습용 `X / y` 생성

## 원본 `.mat` 데이터 구조

각 `.mat` 파일의 최상위에는 보통 아래 키가 있다.

- `batch`
- `batch_date`

`batch` 안에는 여러 배터리 셀이 들어 있고, 각 셀은 대체로 아래 구조를 가진다.

- `Vdlin`
- `barcode`
- `channel_id`
- `cycle_life`
- `cycles`
- `policy`
- `policy_readable`
- `summary`

## Canonical Dataset 구조

출력 경로 예시:

- `data/canonical/<batch_name>/cell_000.npz`
- `data/canonical/<batch_name>/_index.npz`

### 1. Cell 파일(`cell_XXX.npz`)

각 셀 파일은 아래 key들을 가진다.

#### 메타데이터

- `source_batch`
- `cell_id`
- `cycle_life`
- `policy`
- `policy_readable`
- `barcode`
- `channel_id`
- `cycle_count`
- `Vdlin`

설명:

- `source_batch`: 원본 배치 파일 이름
- `cell_id`: 배치 내부 셀 인덱스
- `cycle_life`: EOL(80% SOH) 도달 시 총 cycle 수
- `policy`: 원본 충전 정책 코드
- `policy_readable`: 사람이 읽기 쉬운 충전 정책 문자열
- `barcode`: 셀 식별 바코드
- `channel_id`: 실험 채널 ID
- `cycle_count`: 저장된 총 cycle 개수
- `Vdlin`: 방전 용량 곡선을 정렬할 때 사용하는 기준 전압 축

#### Summary 피처

key prefix: `summary__`

현재 확인된 summary key:

- `summary__cycle`
- `summary__QDischarge`
- `summary__QCharge`
- `summary__IR`
- `summary__Tmax`
- `summary__Tavg`
- `summary__Tmin`
- `summary__chargetime`

설명:

- `summary__cycle`: cycle 번호
- `summary__QDischarge`: cycle별 방전 용량
- `summary__QCharge`: cycle별 충전 용량
- `summary__IR`: cycle별 내부저항
- `summary__Tmax`: cycle별 최고 온도
- `summary__Tavg`: cycle별 평균 온도
- `summary__Tmin`: cycle별 최저 온도
- `summary__chargetime`: cycle별 충전 시간

형태:

- 모두 `1D ndarray`
- 길이는 보통 해당 셀의 cycle 수와 같음

#### Cycle 시계열 피처

key prefix: `cycles__`

현재 확인된 cycle-series key:

- `cycles__index`
- `cycles__I`
- `cycles__Qc`
- `cycles__Qd`
- `cycles__Qdlin`
- `cycles__T`
- `cycles__Tdlin`
- `cycles__V`
- `cycles__discharge_dQdV`
- `cycles__t`

설명:

- `cycles__index`: cycle 번호 목록
- `cycles__I`: cycle 내 전류 시계열
- `cycles__Qc`: cycle 내 누적 충전 용량 시계열
- `cycles__Qd`: cycle 내 누적 방전 용량 시계열
- `cycles__Qdlin`: 정렬된 전압 축에서의 방전 용량 곡선
- `cycles__T`: cycle 내 온도 시계열
- `cycles__Tdlin`: 정렬된 전압 축에서의 온도 곡선
- `cycles__V`: cycle 내 전압 시계열
- `cycles__discharge_dQdV`: 방전 구간의 `dQ/dV` 곡선
- `cycles__t`: cycle 내 시간 시계열

형태:

- `cycles__index`는 `1D ndarray`
- 나머지는 `dtype=object` 인 `1D ndarray`
- 각 원소가 “한 cycle의 시계열 배열”이다

즉:

- `cell["cycles__I"][0]` → 1번째 cycle의 전류 배열
- `cell["cycles__V"][10]` → 11번째 cycle의 전압 배열

### 2. 인덱스 파일(`_index.npz`)

배치 단위 셀 목록을 담는다.

포함 key:

- `source_batch`
- `cell_id`
- `cycle_life`
- `policy`
- `policy_readable`
- `cell_path`

용도:

- 해당 배치에 어떤 셀이 있는지 확인
- 타깃(`cycle_life`) 유무 확인
- 셀 파일 경로 조회

## 현재 확인된 전체 피처 그룹

모델링 후보 관점에서 보면, 현재 데이터는 아래 3그룹으로 나눌 수 있다.

### A. 셀 메타 피처

- `policy`
- `policy_readable`
- `channel_id`
- `barcode`
- `cycle_count`

### B. Cycle-level summary 피처

- `QDischarge`
- `QCharge`
- `IR`
- `Tmax`
- `Tavg`
- `Tmin`
- `chargetime`

이 그룹은 탭ुल러 모델에 가장 바로 쓰기 쉽다.

### C. Raw cycle-series 피처

- `I`
- `Qc`
- `Qd`
- `Qdlin`
- `T`
- `Tdlin`
- `V`
- `discharge_dQdV`
- `t`

이 그룹은 feature engineering 또는 sequence model 입력 후보이다.

## 현재 기본 feature dataset 생성 규칙

`build_feature_dataset.py`는 canonical dataset에서 아래 규칙으로 학습용 데이터를 만든다.

기본 config:

- `early_cycles = 100`
- `target_key = "cycle_life"`
- `include_sequence = true`
- `include_policy_one_hot = false`
- `summary_features = ["QDischarge", "QCharge", "IR", "Tmax", "Tavg", "Tmin", "chargetime"]`
- `aggregations = ["mean", "std", "min", "max", "first", "last", "delta", "slope"]`

즉 각 셀에 대해:

1. 초기 100 cycle의 summary 값을 그대로 펼친 sequence 피처 생성
예:

- `QDischarge_c001`
- `QDischarge_c002`
- ...
- `QDischarge_c100`

2. 같은 값에 대해 집계 피처 생성
예:

- `QDischarge_mean`
- `QDischarge_std`
- `QDischarge_min`
- `QDischarge_max`
- `QDischarge_first`
- `QDischarge_last`
- `QDischarge_delta`
- `QDischarge_slope`

이 규칙이 모든 summary feature에 반복 적용된다.

## 현재 확인된 feature dataset 예시

전체 `.mat` 파일에 대해 canonical dataset을 생성한 결과:

- `2017-05-12_batchdata_updated_struct_errorcorrect`: `46 cells`, `cycle_life 유효 46`
- `2018-02-20_batchdata_updated_struct_errorcorrect`: `47 cells`, `cycle_life 유효 39`
- `2018-04-03_varcharge_batchdata_updated_struct_errorcorrect`: `2 cells`, `cycle_life 유효 0`
- `2018-04-12_batchdata_updated_struct_errorcorrect`: `46 cells`, `cycle_life 유효 44`
- 전체 합계: `141 cells`
- `cycle_life` 유효 샘플 합계: `129`

검증 데이터:

- canonical source: `data/canonical_test2/2018-02-20_batchdata_updated_struct_errorcorrect`
- 생성 결과: `data/features_test2`

확인 결과:

- 전체 셀 수: `47`
- `cycle_life`가 있는 셀 수: `39`
- 생성된 학습 샘플 수: `39`
- 생성된 피처 수: `756`

`756`의 구성:

- `7개 summary feature × 100개 sequence 피처 = 700`
- `7개 summary feature × 8개 aggregate 피처 = 56`
- 총합 `756`

## 권장 운영 방식

피처가 아직 확정되지 않았다면 아래 순서를 권장한다.

1. `.mat -> canonical dataset`은 한 번만 수행
2. 이후 실험은 `feature_config.json`만 바꿔서 반복

이 방식의 장점:

- `.mat` 대용량 파일을 반복해서 다시 읽지 않아도 됨
- summary 기반 피처와 raw cycle 기반 피처 실험을 분리 가능
- regression / classification 모두 같은 canonical dataset을 재사용 가능

## 다음 후보 작업

아래 중 하나를 다음 단계로 진행하면 된다.

- summary 기반 후보 피처셋 3안 정의
- raw cycle-series(`Qdlin`, `dQ/dV`) 기반 파생 피처 추가
- regression용 config와 classification용 config 분리
- train/validation split 모듈 추가

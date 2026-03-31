# ds_project

# ESS 배터리 수명 예측

ESS 배터리 데이터를 기반으로 초기 사이클 정보만을 활용하여 배터리 수명을 예측하는 모델을 구축한 프로젝트이다.

**Batch 간 분포 차이(Distribution Shift)**를 고려한 외부 데이터 일반화 성능 확보를 목표로 실험을 설계하였다.

---

## 프로젝트 개요

데이터셋 : MIT-Stanford Battery Dataset (Severson et al., Nature Energy 2019)
학습 데이터 : Batch 1 (2017-05-12)
평가 데이터 : Batch 2 (2018-02-20)
태스크 : Regression (Cycle Life 예측)

---

## 파일 구조

```
.
├── analysis
│   └── experiment_dashboard.ipynb
├── data
│   └── README.md
├── EDA
│   ├── 01-ESSHealth_cratch_EDA1,4.ipynb
│   ├── 01-ESSHealth-scratch-EDA2,5.ipynb
│   └── 01-ESSHealth-scratct_EDA3.ipynb
├── preprocessing
│   ├── build_canonical_dataset.py
│   ├── build_experiment_split.py
│   ├── build_feature_dataset.py
│   ├── common.py
│   ├── experiment_tracker.py
│   ├── feature_config.example.json
│   ├── feature_config.reduced_summary.json
│   ├── train_elastic_net.py
│   └── train_linear_regression.py
├── requirements.txt
├── Severson_NatureEnergy_2019.pdf
└── submission_compromise11
    ├── README.md
    ├── requirements.txt
    ├── run_compromise11.sh
    └── source
        ├── build_dataset.py
        ├── common.py
        ├── feature_config.compromise11.json
        ├── logger.py
        ├── select_feature.py
        ├── split.py
        └── train_elastic_net.py

## 7 directories, 27 files
```

## 환경 설정

```bash
git clone https://github.com/kareawe/ds_project.git
cd ds_project
pip install -r requirements.txt
```

---

## EDA

### 1. Cycle Life 분포

- **분포 형태 및 장단수명 비율 요약**
  - Batch별로 평균과 분산이 다르게 나타나며, 장수명/단수명 비율도 상이함

- **핵심 발견**
  Batch별로 수명 분포가 크게 다르게 나타났으며, 배터리 수명은 단일 분포가 아닌 충전 조건과 구조 차이에 의해 구분되는 집단 구조를 가진다.

---

### 2. 열화 곡선 분석

- **장수명 vs 단수명 셀의 열화 속도 차이**
  - 장수명 셀은 완만한 감소, 단수명 셀은 빠른 감소 패턴

- **Knee point 존재 여부 및 발생 시점**
  - 대부분의 셀에서 특정 시점 이후 급격한 열화가 발생하는 knee point 확인

- **핵심 발견**
  방전 용량은 cycle이 증가할수록 감소하지만 knee point 이후 급격히 열화가 가속되는 패턴이 나타나며, 열화 전환 시점이 수명을 결정하는 핵심 요인이다.

---

### 3. ΔQ(V) 곡선 분석

- **Cycle 100 - Cycle 10 차이 곡선 형태**
  - 초기 대비 전압 구간별 용량 변화 패턴이 셀마다 다르게 나타남

- **장수명 vs 단수명 셀 비교**
  - 장수명 셀은 변화가 완만하고, 단수명 셀은 특정 구간에서 급격한 변화 발생

- **핵심 발견**
  초기 사이클 ΔQ(V)에서 장수명과 단수명 간 변화 크기와 형태 차이가 나타나 수명과 유의미한 관계를 보이지만, 구조 개선(batch3)에서는 설명력이 감소하므로 단독 변수보다 구조적 요인과 함께 고려해야 하는 핵심 feature이다.

---

### 4. 충전 속도(C-rate)와 수명의 관계

- **충전 프로토콜별 평균 수명 비교**
  - 일부 Batch에서는 충전 속도가 빠를수록 수명이 감소하는 경향
  - 일부 Batch에서는 특정 구간에서 최적 충전 속도 존재

- **핵심 발견**
  충전 속도(C-rate)는 수명과 관계가 있으나 배치에 따라 단조 감소 또는 최적 구간이 존재하는 비선형 패턴을 보이므로, 단순 속도보다 충전 프로토콜 전체가 중요하다.

---

### 5. 상관관계 분석

- **어떤 신호가 수명과 연관되어 있는가**
  - ΔQ(V), 충전 시간, 온도, 초기 충전 특성 등이 수명과 유의미한 상관관계를 보임
  - 특히 초기 충전 시간(`chargetime_cycle2`, `chargetime_cycle50`)이 높은 상관성을 나타냄

- **핵심 발견**
  수명은 ΔQ, 온도, 충전시간 등 다양한 변수와 연관되지만 Batch 2, 3에서 구조 변화가 상관관계를 바꿀 정도로 크게 작용하므로, cell 구조가 가장 지배적인 결정 요인이다.

## EDA 핵심 요약

- 배터리 수명은 Batch별로 서로 다른 분포를 가지는 집단 구조를 보임
- 열화는 선형이 아닌 knee point 기반 비선형 패턴
- ΔQ(V), 내부저항(IR), 충전시간, 온도가 주요 설명 변수
- 동일 feature라도 Batch에 따라 의미가 달라지는 distribution shift
  존재

---

## Modeling

### 피처 엔지니어링 전략

EDA 결과를 바탕으로 **Batch 간 분포 차이(Distribution Shift)**를 고려한 feature 설계를 수행하였다.

- ΔQ(V), IR, 충전시간, 온도가 수명과 주요하게 연관됨
- 단일 값보다 변화량(delta)과 추세(slope)가 더 중요한 정보 제공
- Batch에 따라 동일 feature의 의미가 달라지는 현상 확인

이에 따라 다음과 같은 전략을 적용하였다.

- last + delta + slope 동시 사용 (구조 변화 대응)
- ΔQ(V) 기반 feature 유지 (초기 열화 패턴 반영)
- 물리적 의미 기반 변수 선택 (IR, 온도, 충전시간)
- 중복 feature 최소 제거 (generalization 유지)

---

### Multicollinearity (VIF)

일부 변수(`QDischarge`, `IR`, `chargetime`의 last/delta 조합)에서 VIF 값이 높게 나타났다.

이는 Train Batch에서 해당 변수들이 거의 동일한 값을 가지는 구조적 특성으로 인해 발생한 것이다.

그러나 External Batch에서는 이러한 관계가 유지되지 않으며,
동일 feature라도 Batch에 따라 의미와 역할이 달라지는 현상이 확인되었다.

따라서 본 프로젝트에서는 단순히 다중공선성을 제거하는 대신,

→ **구조적 collinearity와 generalization 간 trade-off를 고려하여 feature를 유지하는 전략을 선택하였다**

즉, Train 데이터 기준의 통계적 안정성보다
**External 데이터에서의 성능 유지(일반화)를 우선하였다.**

---

### Feature Set

### Input (X)

초기 사이클에서 추출한 **열화 상태, 변화 패턴, 물리적 특성 변수**

- QDischarge_last, QDischarge_delta, QDischarge_slope
- IR_std, IR_last, IR_delta, IR_slope
- Tavg_mean
- chargetime_last, chargetime_delta
- Qdlin_delta_c100_c010_logvar

### Output (y)

배터리 전체 수명(cycle life)

수명 분포의 비대칭(long-tailed)을 완화하기 위해 로그 변환 후 학습하며, 예측 결과는 원래 scale로 복원하여 평가

- Target: cycle_life
- 학습 시: log(cycle_life)

---

## ESS 도메인 해석

### 1. 실제 BESS 적용 시 활용

- 초기 셀 스크리닝
    → 초기 변화 패턴으로 저수명 셀을 조기에 걸러냄

- 충전 전략 개선
    → 충전 시간/패턴을 기반으로 수명에 유리한 운영 방식 설계

- 이상 셀 탐지
    → 내부저항 변화로 열화가 빠른 셀을 조기 감지

- 운영 효율화
    → 예측 수명에 따라 셀 역할을 나눠 시스템 안정성 향상

간단 해석:
초기 몇 사이클 데이터만으로 배터리의 전체 수명을 미리 예측하여, 선제적인 운영과 관리가 가능하다.

---

### 2. 한계 및 개선 방향

- 특정 batch 기반 → 다양한 데이터로 일반화 필요
- 실제 환경 변수 부족 → 온도, 사용 패턴 등 추가 필요
- 다중공선성 존재 → 해석 안정성 제한
- 실시간 적용 부족 → BMS 연동 필요

간단 해석:
현재 모델은 실험 데이터 기반이므로, 실제 ESS 환경에 적용하려면 데이터 확장과 시스템 연동이 필요하다.

## ESS 도메인 해석

분석 결과를 실제 ESS 운영 관점에서 해석

- 이 모델을 실제 BESS에 적용한다면 어떤 의사결정에 활용 가능한가?
- 어떤 한계가 있으며, 실 배포를 위해 추가로 필요한 것은 무엇인가?

## 참고문헌

- Severson et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. _Nature Energy_, 4, 383–391.

## 팀 구성

- 김민재 : EDA(1,4), 데이터 전처리, 성능평가
- 박하정 : EDA(3), 피처엔지니어링
- 윤민후 : EDA(2,5) 모델 개발, 성능 평가

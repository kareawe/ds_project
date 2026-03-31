# Battery Dataset Structure Overview

이 데이터셋은 배터리 셀 단위로 구성되어 있으며, 각 셀마다 최종 수명 정보, 충전 조건, 사이클별 요약 데이터, 그리고 각 사이클 내부의 상세 시계열 데이터가 함께 저장되어 있다.

구조적으로 보면, 먼저 셀 단위의 기본 정보가 존재하고, 그 아래에 사이클별 요약 정보(`summary`)와 각 사이클 내부의 상세 측정값(`cycles`)이 연결된다.

- `summary`는 각 사이클에서의 방전용량, 충전용량, 내부저항, 온도, 충전시간처럼 비교적 압축된 형태의 지표를 담고 있어 전체적인 열화 추세를 파악하는 데 적합하다.
- `cycles`는 시간, 전압, 전류, 온도, 용량 등의 원본 시계열 데이터를 포함하므로, 특정 사이클 내부에서의 충전/방전 곡선 형태나 세부 패턴 분석에 활용할 수 있다.

특히 이 데이터셋에서는 `cycle_life`가 예측 대상인 정답값이며, `policy`, `summary`, `Qdlin`, `dQ/dV` 등의 정보는 수명 예측이나 열화 특성 분석을 위한 주요 입력 후보로 활용될 수 있다.

## 1. Overall Structure

배터리 셀 1개는 다음과 같은 구조를 가진다.

```text
Battery Cell
├── cycle_life        # 배터리 총 수명 (정답)
├── policy            # 충전 조건
├── summary           # 사이클별 요약 데이터
└── cycles            # 사이클 내부 상세 데이터
```

---

## 2. Summary: Cycle-level Aggregated Data

`summary`는 각 사이클을 하나의 요약값으로 정리한 데이터이다.  
배터리의 전체 열화 추세를 확인하거나, 사이클별 변화량을 기반으로 feature를 만들 때 주로 사용한다.

| 변수명       | 타입    | 의미                       | 활용           |
| ------------ | ------- | -------------------------- | -------------- |
| `cycle`      | ndarray | 사이클 번호 (1, 2, 3, ...) | 시간 축        |
| `QDischarge` | ndarray | 방전 용량                  | 열화 분석      |
| `QCharge`    | ndarray | 충전 용량                  | 효율 분석      |
| `IR`         | ndarray | 내부 저항                  | 열화 지표      |
| `Tavg`       | ndarray | 평균 온도                  | 열 영향 분석   |
| `Tmax`       | ndarray | 최대 온도                  | 이상 탐지      |
| `Tmin`       | ndarray | 최소 온도                  | 안정성 분석    |
| `chargetime` | ndarray | 충전 시간                  | 충전 특성 분석 |

---

## 3. Cycles: In-cycle Detailed Time-series Data

`cycles`는 각 사이클 내부에서 측정된 원본 시계열 데이터이다.  
즉, 한 사이클이 진행되는 동안 시간에 따라 전압, 전류, 온도, 용량이 어떻게 변하는지를 담고 있다.

| 변수명 | 타입    | 의미      | 특징      | 활용        |
| ------ | ------- | --------- | --------- | ----------- |
| `t`    | ndarray | 시간      | 길이 가변 | 원본 데이터 |
| `V`    | ndarray | 전압      | 시간 기준 | 원본 데이터 |
| `I`    | ndarray | 전류      | 시간 기준 | 원본 데이터 |
| `T`    | ndarray | 온도      | 시간 기준 | 원본 데이터 |
| `Qd`   | ndarray | 방전 용량 | 시간 기준 | 원본 데이터 |
| `Qc`   | ndarray | 충전 용량 | 시간 기준 | 원본 데이터 |

---

## 4. Curve-related Variables

이 데이터셋에서는 단순 요약값뿐 아니라, 곡선 형태 자체를 분석할 수 있는 변수들도 중요하다.

| 변수명  | 타입            | 설명                                 | 활용                         |
| ------- | --------------- | ------------------------------------ | ---------------------------- |
| `Vdlin` | ndarray (1000,) | 공통 전압 축                         | 곡선 비교 기준               |
| `Qdlin` | ndarray         | 전압 축에 맞춰 정렬된 방전 용량 곡선 | 곡선 비교, ΔQ 분석           |
| `dQ/dV` | ndarray         | 전압 대비 용량 변화율 패턴           | 열화 패턴 분석, feature 추출 |

---

## 5. Batch Dataset Characteristics

### Batch 1

- 안정적, 균일한 수명 분포 (≈845 cycle)
- 고속 충전 → 온도 상승 → 수명 감소
- ΔQ 기반 feature의 설명력 높음

### Batch 2

- 단수명 셀 비율 높음 (≈60%)
- 고속 충전 → 발열 및 IR 증가 → 수명 급감
- 구조(newstructure) 영향 일부 존재

### Batch 3

- Batch 간 분포 차이 존재 (distribution shift)
- ΔQ 기반 feature 변별력 감소
- 구조적 특성 추가 고려 필요

---

## 6. Modeling Strategy (Classification)

### Target

- Long-life: >
- Short-life: <

### Feature

-

### Data Split

- Train: Batch 1
- Test: Batch 2
- Additional: Batch 3

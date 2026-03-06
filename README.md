# QuarterNet v1

QuarterNet v1은 Binance Futures `BTCUSDT` 15분봉 데이터를 이용해 미래 24시간의 방향을 예측하는 시계열 딥러닝 프로젝트입니다. 현재 코드 기준의 핵심 태스크는 "미래 96개 캔들의 `lnCO` 누적합이 양수인지 음수인지"를 맞히는 이진 분류이며, 모델은 샘플당 단일 logit을 출력합니다.

이 README는 현재 저장소의 실제 코드 경로인 `config.py`, `train.py`, `eval.py`, `core/net.py`, `data/dataset.py`, `data/apicalling.py`를 기준으로 작성되었습니다. 과거의 96-step 회귀형 설명이나 오래된 아키텍처 수치는 현재 기본 경로를 설명하지 않습니다.

## 한눈에 보기

| 항목 | 현재 구현 |
|---|---|
| 시장 | Binance Futures `BTCUSDT` |
| 데이터 간격 | `15m` |
| 과거 입력 길이 | `9,600` 스텝 (`100일`) |
| 미래 구간 길이 | `96` 스텝 (`24시간`) |
| 입력 텐서 | `(B, 9600, 22)` |
| 데이터셋 타깃 저장 형식 | `(B, 96, 5)` |
| 실제 학습 라벨 | `sum(lnCO_0..95) > 0` |
| 모델 출력 | 단일 direction logit `(B,)` |
| 손실 함수 | `BCEWithLogitsLoss` |
| 평가 지표 | 방향 정확도(`accuracy`) 중심 |
| 핵심 스크립트 | `train.py`, `eval.py` |
| 최고 체크포인트 기준 | `eval accuracy` 최대 |

## 설치

### Python 패키지

의존성은 `requirements.txt`에 정의되어 있습니다.

```bash
pip install -r requirements.txt
```

현재 요구 패키지는 다음과 같습니다.

```text
requests>=2.32
pandas>=2.2
numpy>=2.2
h5py>=3.11
torch>=2.9
matplotlib>=3.9
```

### 시각화 폰트

학습 중 생성되는 시각화 이미지는 한글 폰트가 있으면 더 자연스럽게 렌더링됩니다. 코드에서는 아래 순서로 폰트를 탐색합니다.

- `NanumGothic`
- `Malgun Gothic`
- `AppleGothic`
- `DejaVu Sans`

Windows는 보통 `Malgun Gothic`, macOS는 `AppleGothic`이 자동으로 잡히고, Linux에서는 필요하면 아래처럼 `NanumGothic`을 설치할 수 있습니다.

```bash
apt-get update && apt-get install -y fonts-nanum
rm -rf "$(python -c 'import matplotlib; print(matplotlib.get_cachedir())')"
```

## 빠른 시작

가장 빠른 실행 경로는 아래와 같습니다.

```bash
# 1) 의존성 설치
pip install -r requirements.txt

# 2) 처음부터 학습 시작
python train.py --restart

# 3) 평가
python eval.py --checkpoint checkpoints/best_export.pt --samples 1000 --batch-size 8
```


```bash
git clone https://github.com/Tom-Heo/quarter_v1.git
cd quarter_v1
pip install -r requirements.txt
apt-get update && apt-get install -y fonts-nanum
rm -rf "$(python -c 'import matplotlib; print(matplotlib.get_cachedir())')"
python train.py --restart
```

추가로 알아둘 점은 다음과 같습니다.

- `python train.py`는 기본적으로 `resume` 동작을 시도합니다.
- 데이터셋 HDF5가 없으면 `train.py`와 `eval.py`가 자동으로 생성합니다.
- `eval.py`는 `last.pt`나 `best.pt` 같은 전체 학습 체크포인트가 아니라 `last_export.pt`, `best_export.pt` 같은 export 가중치만 지원합니다.

데이터셋만 미리 만들어 두고 싶다면 아래 명령을 사용할 수 있습니다.

```bash
python data/dataset.py
```

## 프로젝트가 푸는 문제

현재 프로젝트는 "미래 캔들 경로 전체를 회귀"하는 기본 경로가 아닙니다. 현재 학습/평가 경로는 아래의 방향 분류 문제를 풉니다.

```text
label = 1 if sum_{t=0}^{95} ln(C_t / O_t) > 0 else 0
prediction = 1 if logit > 0 else 0
P(up) = sigmoid(logit)
```

`eval.py`의 해석 기준에 따르면,

```text
C_95 / O_0 = exp(sum(lnCO_0..95))
```

이므로, 미래 96개 `lnCO`의 누적합 부호만 보면 24시간 후 방향을 판정할 수 있습니다.

중요한 점은 데이터셋이 여전히 더 풍부한 타깃을 저장한다는 것입니다.

- 저장 타깃은 `96 x 5` 시퀀스입니다.
- 현재 모델은 그중 첫 채널 `lnCO`만 사용해 방향 라벨을 만듭니다.
- 따라서 데이터셋 저장 형식과 실제 모델 출력 계약은 다릅니다.

## 입출력 계약

### 입력

- 입력 텐서 shape: `(B, 9600, 22)`
- 입력은 과거 100일의 15분봉 피처 시퀀스입니다.
- 입력 피처는 split 단위 mean/std로 z-score 정규화되어 HDF5에 저장됩니다.

### 모델 출력

- 출력 shape: `(B,)`
- 각 원소는 단일 direction logit입니다.
- `logit > 0`이면 상승 예측, `logit <= 0`이면 하락 예측입니다.

### 데이터셋 타깃 저장 형식

- 저장 타깃 shape: `(B, 96, 5)`
- 타깃은 정규화되지 않은 원시 log-return 값입니다.
- 현재 학습 라벨은 이 중 `y[..., 0]`만 사용합니다.

### 저장 타깃 5개

| 인덱스 | 이름 | 정의 |
|---|---|---|
| 0 | `lnCO` | `ln(Close / Open)` |
| 1 | `lnHO` | `ln(High / Open)` |
| 2 | `lnLO` | `ln(Low / Open)` |
| 3 | `lnCH` | `ln(Close / High)` |
| 4 | `lnCL` | `ln(Close / Low)` |

## 데이터 파이프라인

### 데이터 소스

현재 구현이 수집하는 데이터는 정확히 세 가지입니다.

| 소스 | 용도 | 비고 |
|---|---|---|
| `klines` | OHLCV, 거래 수(`trades`), `taker_buy_vol` | Binance Futures klines |
| `funding_rate` | 외생 피처 | Binance funding rate |
| `basis` | 외생 피처 | `CURRENT_QUARTER` basis |

기존 문서에서 보이던 `long_short_ratio`, `open_interest`는 현재 코드 경로에 포함되어 있지 않습니다.

### 정렬과 전처리

데이터셋 생성 흐름은 아래와 같습니다.

```text
Binance API 수집
-> timestamp 기준 정렬
-> funding_rate / basis join
-> forward fill
-> dropna
-> feature engineering
-> feature z-score 정규화
-> HDF5 저장
-> QuarterDataset에서 슬라이딩 윈도우로 샘플 구성
```

세부 동작은 다음과 같습니다.

1. `klines`를 기준 테이블로 사용합니다.
2. `funding_rate`, `basis`를 timestamp 인덱스로 join합니다.
3. 외생 시계열은 `ffill()`로 메우고, 이후 `dropna()`를 적용합니다.
4. 로그수익률과 주기 인코딩을 포함한 피처를 생성합니다.
5. feature-wise mean/std로 정규화한 뒤 `features`, `targets` 전체 배열을 HDF5에 저장합니다.
6. 실제 샘플 생성은 HDF5 저장 시점이 아니라 `QuarterDataset.__getitem__()`에서 슬라이싱으로 수행합니다.

### 기간 분할

`config.py` 기준의 기본 데이터 기간은 아래와 같습니다.

| 구분 | 시작 | 종료 |
|---|---|---|
| 학습 | `2020-01-01` | `2025-06-30` |
| 평가 | `2025-07-01` | `2025-12-31` |

### 입력 피처 22개

| # | 이름 | 범주 | 정의 또는 의미 |
|---|---|---|---|
| 0 | `f1` | 캔들 형태 | `lnCO` |
| 1 | `f2` | 캔들 형태 | `clip(lnHO * lnLO * lnCH * lnCL / body_safe^3)` |
| 2 | `f3` | 캔들 형태 | `clip(lnHO * lnCH / body_safe)` |
| 3 | `f4` | 캔들 형태 | `clip(lnLO * lnCL / body_safe)` |
| 4 | `lnHO` | 가격 관계 | `ln(High / Open)` |
| 5 | `lnLO` | 가격 관계 | `ln(Low / Open)` |
| 6 | `lnCH` | 가격 관계 | `ln(Close / High)` |
| 7 | `lnCL` | 가격 관계 | `ln(Close / Low)` |
| 8 | `lnHL` | 가격 관계 | `ln(High / Low)` |
| 9 | `log_volume` | 거래 활동 | `clip(ln(V_t / V_{t-1}))` |
| 10 | `log_trades` | 거래 활동 | `clip(ln(T_t / T_{t-1}))` |
| 11 | `taker_buy_ratio` | 거래 활동 | `taker_buy_vol / volume`, volume이 매우 작으면 `0.5` |
| 12 | `funding_rate` | 외생 변수 | Binance funding rate 원본값 |
| 13 | `basis` | 외생 변수 | Binance current quarter basis 원본값 |
| 14 | `sin_hour` | 시간 인코딩 | 시간대 사인 |
| 15 | `cos_hour` | 시간 인코딩 | 시간대 코사인 |
| 16 | `sin_dow` | 시간 인코딩 | 요일 사인 |
| 17 | `cos_dow` | 시간 인코딩 | 요일 코사인 |
| 18 | `sin_year` | 시간 인코딩 | 연중 위치 사인 |
| 19 | `cos_year` | 시간 인코딩 | 연중 위치 코사인 |
| 20 | `sin_fund` | 시간 인코딩 | 8시간 펀딩 사이클 사인 |
| 21 | `cos_fund` | 시간 인코딩 | 8시간 펀딩 사이클 코사인 |

추가 설명:

- `body_safe`는 `lnCO`가 0에 가까울 때 분모가 터지지 않도록 `EPSILON`으로 보호한 값입니다.
- `log_volume`, `log_trades`, `f2`, `f3`, `f4`는 `CLIP_BOUND`로 클리핑됩니다. 기본값은 `10.0`입니다.
- 로그수익률 계산 때문에 첫 행이 제거되며, 모든 피처와 타깃은 그 이후 길이에 맞춰 정렬됩니다.

### HDF5 저장 형식

현재 구현은 윈도우별 샘플을 미리 저장하지 않고, 전체 시계열 배열을 HDF5에 저장합니다.

| 항목 | 형식 | 설명 |
|---|---|---|
| `features` | `(T, 22)` `float32` | 정규화된 입력 피처 전체 |
| `targets` | `(T, 5)` `float32` | 정규화되지 않은 타깃 전체 |
| `attrs["features"]` | 문자열 배열 | 현재 피처 스키마 |
| `attrs["target_features"]` | 문자열 배열 | 현재 타깃 스키마 |
| `attrs["mean"]` | 실수 배열 | 피처 정규화 mean |
| `attrs["std"]` | 실수 배열 | 피처 정규화 std |
| `attrs["seq_len"]` | 정수 | 입력 길이 |
| `attrs["target_len"]` | 정수 | 미래 길이 |
| `attrs["stride"]` | 정수 | 슬라이딩 윈도우 stride |
| `attrs["n_samples"]` | 정수 | 샘플 수 |

샘플 수는 아래 식으로 계산됩니다.

```text
n_samples = (T - seq_len - target_len) // stride + 1
```

현재 기본값은 `seq_len=9600`, `target_len=96`, `stride=1`입니다.

## 모델 구조

### 전체 흐름

```text
Input (B, 9600, 22)
-> EmbeddingBlock
-> FFNBlock
-> learned CLS token 1개를 시퀀스 뒤에 append
-> QuarterBlock x64
-> 마지막 CLS 토큰 추출
-> FFNBlock
-> Linear(2048 -> 1)
-> direction logit (B,)
```

현재 구조에서 CLS 토큰은 시퀀스 끝에 붙습니다. 따라서 causal attention 하에서 마지막 CLS 토큰은 전체 과거 입력 컨텍스트를 모두 볼 수 있고, 이 토큰 하나가 방향 요약 벡터 역할을 합니다.

### 기본 사양

| 항목 | 값 |
|---|---|
| `d_model` | `2048` |
| `num_heads` | `16` |
| `head_dim` | `128` |
| `QuarterBlock` 수 | `64` |
| FFN hidden dim | `4096` (`d_model * 2`) |
| CLS 토큰 수 | `1` |
| 출력 차원 | `1` |
| RoPE base | `10000.0` |
| 최대 시퀀스 길이 | `9601` (`9600 + 1`) |
| 어텐션 | causal scaled dot-product attention |
| 체크포인팅 | block별 gradient checkpointing 사용 |

### 커스텀 모듈

- `HeLU` / `HeLUseq`: 비대칭 GELU 계열의 학습 가능한 활성화 모듈입니다.
- `HeoGate`: 잔차 경로와 변환 경로를 학습 가능한 게이트로 섞습니다. 형태는 `(alpha * x + beta * residual) / 2`입니다.
- `Heoptimizer`: `AdamW` 기반 옵티마이저로, `HeLU`/`HeoGate` 계열 파라미터에 별도 learning rate scaling과 zero weight decay를 적용합니다.
- `HeoLoss`: 코드에는 남아 있지만, 현재 기본 학습 경로에서는 사용하지 않습니다.

### 현재 구조의 제약

`QuarterNet`은 현재 명시적으로 아래 둘만 지원합니다.

- `NUM_CLS_TOKENS = 1`
- `NUM_DIRECTION_OUTPUTS = 1`

다른 값으로 생성하려고 하면 `ValueError`를 발생시킵니다.

## 학습

### 실행 방법

```bash
# 기본 동작: 이어서 학습 시도
python train.py

# 명시적 resume
python train.py --resume

# 처음부터 새로 학습
python train.py --restart
```

`python train.py`와 `python train.py --resume`은 체크포인트가 있으면 이어서 학습하고, 없으면 처음부터 시작합니다. `--restart`는 기존 체크포인트를 삭제하지는 않지만 로드를 건너뜁니다.

### 학습 하이퍼파라미터

현재 기본값은 아래와 같습니다.

| 항목 | 값 |
|---|---|
| 배치 크기 | `1` |
| 옵티마이저 | `Heo.Heoptimizer` |
| 학습률 | `1e-4` |
| Weight Decay | `1e-4` |
| EMA decay | `0.999` |
| 워밍업 시작 배율 | `1e-7` |
| 스케줄러 | `SequentialLR(LinearLR -> ExponentialLR)` |
| Exponential gamma | `0.999998` |
| 손실 함수 | `BCEWithLogitsLoss` |
| 로그 주기 | `8` step |
| 내부 평가 주기 | `256` step |
| 내부 평가 샘플 수 | `256` |
| 최고 모델 기준 | `eval accuracy` |

### 학습 루프 동작

- 학습은 `while True` 루프로 돌며, 수동 중단 전까지 계속됩니다.
- 각 step마다 `EMA`를 갱신합니다.
- `LOG_INTERVAL=8`마다 학습 BCE, 학습 accuracy, learning rate, step 속도를 기록합니다.
- 같은 시점에 EMA 파라미터로 시각화를 생성해 `outputs/step_XXXXXX.png`에 저장합니다.
- `EVAL_INTERVAL=256`마다 랜덤 `256`개 eval 샘플에 대해 BCE와 accuracy를 계산합니다.
- `best`는 loss가 아니라 accuracy 기준으로 갱신됩니다.
- CUDA를 사용할 수 없으면 경고를 출력하고 CPU 학습으로 자동 전환합니다.

### 내부 평가와 시각화

현재 시각화는 예전의 캔들 복원 그림이 아닙니다. 현재 출력 이미지는 다음을 보여줍니다.

- 정답 `lnCO` 누적 경로
- 예측 방향 (`UP` / `DOWN`)
- `sigmoid(logit)`으로 계산한 상승 확률
- 샘플 인덱스

즉, 시각화도 현재 분류 태스크에 맞게 바뀌어 있습니다.

### 체크포인트 재개 동작

`train.py`는 `last.pt`를 읽어 재개를 시도하지만, 아래 경우에는 새 학습으로 되돌아갑니다.

- 저장된 `task` 마커가 현재 태스크와 다를 때
- 현재 모델 구조와 `state_dict` shape가 맞지 않을 때

이 경우 스크립트는 메시지를 출력하고 처음부터 다시 학습합니다.

## 평가

### 실행 방법

```bash
# 기본 평가
python eval.py

# 옵션 지정 평가
python eval.py --checkpoint checkpoints/best_export.pt --samples 5000 --batch-size 8 --seed 42 --device cuda:0
```

### CLI 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--checkpoint` | `checkpoints/last_export.pt` | 평가할 export 체크포인트 경로 |
| `--samples` | `1000` | 무작위 평가 샘플 수 |
| `--batch-size` | `1` | 추론 배치 크기 |
| `--seed` | `None` | 샘플링 재현용 시드 |
| `--device` | 자동 선택 | `cuda`, `cuda:0`, `cpu` 등 |

### 평가 동작

- 평가용 HDF5가 없으면 자동으로 생성합니다.
- `last_export.pt`가 없고 `best_export.pt`가 있으면 자동으로 fallback합니다.
- 정답 라벨은 `sum(lnCO_0..95) > 0`으로 계산합니다.
- 예측은 `logit > 0`으로 판정합니다.
- 주 지표는 accuracy입니다.

### 중요한 제약

`eval.py`는 전체 학습 체크포인트를 받지 않습니다. 즉, 아래는 실패합니다.

- `checkpoints/last.pt`
- `checkpoints/best.pt`

아래만 지원합니다.

- `checkpoints/last_export.pt`
- `checkpoints/best_export.pt`

## 체크포인트, 로그, 산출물

| 경로 | 내용 | 용도 |
|---|---|---|
| `checkpoints/last.pt` | 전체 학습 상태 | resume 용 |
| `checkpoints/best.pt` | 최고 accuracy 시점의 전체 학습 상태 | resume / 보관 용 |
| `checkpoints/last_export.pt` | EMA가 적용된 모델 가중치만 저장 | 평가 / 추론 용 |
| `checkpoints/best_export.pt` | 최고 accuracy 기준 EMA export 가중치 | 평가 / 추론 용 |
| `logs/train_*.log` | 학습 로그 | 진행 추적 |
| `logs/eval_*.log` | 평가 로그 | 평가 기록 |
| `outputs/step_*.png` | 분류 시각화 이미지 | 학습 중 모니터링 |
| `dataset/*.h5` | train/eval 데이터셋 | 반복 학습용 입력 |

추가 설명:

- `last.pt`와 `best.pt`는 모델만이 아니라 optimizer, scheduler, EMA shadow, epoch, step, best accuracy까지 포함합니다.
- `last_export.pt`와 `best_export.pt`는 EMA 적용 상태에서 저장한 순수 `state_dict`입니다.

## 주요 설정값 안내

자주 보게 되는 설정은 아래와 같습니다.

| 상수 | 기본값 | 의미 |
|---|---|---|
| `FEATURES` | 22개 피처 목록 | 입력 스키마 |
| `SEQ_LEN` | `9600` | 과거 입력 길이 |
| `TARGET_LEN` | `96` | 미래 타깃 길이 |
| `STRIDE` | `1` | 슬라이딩 윈도우 stride |
| `NUM_CLS_TOKENS` | `1` | 모델에 append할 CLS 토큰 수 |
| `NUM_DIRECTION_OUTPUTS` | `1` | 모델 최종 출력 차원 |
| `BATCH_SIZE` | `1` | 학습 배치 크기 |
| `TRAIN_DATASET_START` | `2020-01-01` | 학습 시작일 |
| `TRAIN_DATASET_END` | `2025-06-30` | 학습 종료일 |
| `EVAL_DATASET_START` | `2025-07-01` | 평가 시작일 |
| `EVAL_DATASET_END` | `2025-12-31` | 평가 종료일 |
| `DATASET_DIR` | `dataset` | HDF5 저장 디렉터리 |
| `CHECKPOINT_DIR` | `checkpoints` | 체크포인트 디렉터리 |
| `BINANCE_SYMBOL` | `BTCUSDT` | 수집 심볼 |
| `BINANCE_INTERVAL` | `15m` | 데이터 수집 간격 |

설정을 바꿀 때의 주의점:

- `FEATURES`가 바뀌면 기존 HDF5는 legacy로 간주되어 자동 삭제 후 재생성됩니다.
- 하지만 현재 legacy 검사는 주로 피처 스키마를 기준으로 합니다.
- 따라서 `SEQ_LEN`, `TARGET_LEN`, 태스크 정의, 모델 구조를 바꾼 경우에는 기존 데이터셋과 체크포인트를 수동으로 정리하는 편이 안전합니다.

## 프로젝트 구조

```text
quarter_v1/
├── config.py
├── train.py
├── eval.py
├── inference.py
├── requirements.txt
├── README.md
├── core/
│   ├── block.py
│   ├── heo.py
│   └── net.py
├── data/
│   ├── apicalling.py
│   └── dataset.py
├── dataset/
├── checkpoints/
├── logs/
└── outputs/
```

각 파일의 역할은 대략 아래와 같습니다.

- `config.py`: 데이터, 모델, 경로 관련 상수 정의
- `train.py`: 학습, EMA, 평가, 체크포인트 저장, 시각화
- `eval.py`: export 체크포인트 기반 방향 정확도 평가
- `core/net.py`: 현재 direction classification용 `QuarterNet`
- `core/block.py`: `EmbeddingBlock`, `AttentionBlock`, `FFNBlock`, `QuarterBlock`, `RoPE`
- `core/heo.py`: `HeLU`, `HeoGate`, `Heoptimizer`, `HeoLoss`
- `data/apicalling.py`: Binance 데이터 수집
- `data/dataset.py`: 피처 생성, HDF5 저장, `QuarterDataset`
- `inference.py`: 현재 비어 있는 placeholder 파일

## 제한사항과 트러블슈팅

### 현재 코드 경로의 제한사항

- 현재 기본 태스크는 회귀가 아니라 방향 분류입니다.
- 모델은 단일 CLS 토큰과 단일 direction output만 지원합니다.
- `eval.py`는 export 체크포인트만 지원합니다.
- `train.py` 내부 평가는 매번 전체 eval set이 아니라 랜덤 `256`개 샘플을 사용합니다.
- `eval.py`도 기본적으로 랜덤 `1000`개 샘플을 사용하므로, `--seed`를 주지 않으면 수치가 조금씩 달라질 수 있습니다.
- `inference.py`는 현재 비어 있으므로 공식 추론 진입점으로 문서화하지 않습니다.

### 자주 헷갈리는 점

**Q. 데이터셋 타깃이 `(96, 5)`인데 왜 모델 출력은 `(B,)`인가요?**  
현재 데이터셋은 미래 캔들 정보를 더 풍부하게 저장하지만, 실제 학습/평가 라벨은 `lnCO` 누적 부호 하나만 사용합니다. 즉 저장 포맷과 현재 모델 계약이 분리되어 있습니다.

**Q. 왜 `eval.py`에 `last.pt`를 넣으면 실패하나요?**  
`eval.py`는 전체 학습 상태가 아니라 export된 순수 가중치만 받도록 설계되어 있습니다. `last_export.pt` 또는 `best_export.pt`를 사용해야 합니다.

**Q. 기존 회귀형 체크포인트를 그대로 이어서 쓸 수 있나요?**  
보장되지 않습니다. 현재 모델은 single-CLS, single-logit 구조이므로 이전 회귀형 체크포인트와 shape가 다를 수 있습니다.

**Q. 데이터셋은 언제 다시 만들어야 하나요?**  
피처 스키마가 바뀌면 자동 재생성되지만, `SEQ_LEN`, `TARGET_LEN`, 태스크 정의가 바뀐 경우에는 기존 HDF5를 수동으로 지우고 다시 만드는 편이 안전합니다.

## 현재 상태 요약

현재 저장소의 기본 경로를 한 문장으로 요약하면 다음과 같습니다.

> 과거 100일의 22개 피처 시퀀스를 입력으로 받아, 미래 24시간 동안의 `lnCO` 누적 방향을 single-logit 이진 분류로 예측하고, `BCEWithLogitsLoss`와 accuracy 기준으로 학습/평가하는 QuarterNet 구현입니다.

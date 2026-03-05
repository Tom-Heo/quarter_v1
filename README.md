# QuarterNet v1

Binance BTCUSDT 15분봉 캔들스틱 패턴을 예측하는 시계열 딥러닝 모델.

9,600봉(100일)의 과거 데이터로부터 96봉(24시간) 후의 캔들스틱 형태를 예측한다.

---

## 아키텍처

```
입력 (B, 9600, 9)
       │
       ▼
┌──────────────┐
│ EmbeddingBlock│  9 → 4096 (triple-gated HeLU + cube root)
└──────┬───────┘
       │ (B, 9600, 4096)
       │
       │  ┌─────────────────┐
       ├──┤ CLS tokens (96) │  학습 가능한 위치 토큰
       │  └─────────────────┘
       │
       ▼ (B, 9696, 4096)
┌──────────────┐
│ QuarterBlock │ × 32
│  ├ Attention │  RoPE + Causal SDPA + HeoGate 잔차
│  └ FFN       │  Triple-gated HeLU + cube root + HeoGate 잔차
└──────┬───────┘
       │ (B, 9696, 4096)
       │
       │  마지막 96 토큰 추출
       ▼ (B, 96, 4096)
┌──────────────┐
│  Linear Head │  4096 → 5
└──────┬───────┘
       │
       ▼
출력 (B, 96, 5)
```

### 모델 사양

| 항목 | 값 |
|---|---|
| d_model | 4,096 |
| num_heads | 32 |
| head_dim | 128 |
| QuarterBlock 수 | 32 |
| FFN 은닉 차원 | 8,192 (d_model × 2) |
| 입력 시퀀스 길이 | 9,600 |
| 출력 시퀀스 길이 | 96 |
| 위치 인코딩 | RoPE (base=10,000) |
| 어텐션 | Causal Scaled Dot-Product |
| 활성화 함수 | HeLU (학습 가능한 비대칭 GELU) |
| 잔차 연결 | HeoGate (학습 가능한 α/β 게이트) |
| FFN 게이팅 | Triple gate (gate1 × gate2 × feature) + cube root |

### 핵심 구성 요소

**HeLU** — 학습 가능한 파라미터 `red`, `blue`, `α`, `β`를 가지는 비대칭 활성화 함수. 양수 영역(red·GELU)과 음수 영역(blue·GELU)의 기여를 독립적으로 학습하며, 원본 입력과의 가중 합산을 통해 항등 경로를 보존한다.

**HeoGate** — 잔차 연결에 사용되는 학습 가능 게이트. `(α·x + β·residual) / 2` 형태로, `tanh(√3·w) + 1` 변환을 통해 게이트 값을 `[0, 2]` 범위로 제한한다. 초기값 0에서 시작하여 학습 초기엔 입력과 잔차를 균등 혼합한다.

**Heoptimizer** — AdamW 기반 맞춤형 옵티마이저. HeLU/HeoGate 모듈의 파라미터에는 `lr × lr_scale`의 확대된 학습률을 적용하고 weight decay를 해제한다. 일반 파라미터에는 기본 lr과 weight decay를 적용한다.

**HeoLoss** — Charbonnier loss와 sharp loss의 하이브리드. `|diff| ≤ 0.01` 구간에서는 `log(1 + 100·charbonnier/ε) / 100`으로 작은 오차에 민감하게 반응하고, 그 외 구간에서는 L1 loss로 전환하여 큰 오차에 안정적으로 대응한다.

---

## 입출력 스펙

### 입력 피처 (9개)

| 인덱스 | 이름 | 정의 | 설명 |
|---|---|---|---|
| 0 | f1 | ln(C/O) | 캔들 몸통 (body) |
| 1 | f2 | lnHO·lnLO·lnCH·lnCL / body³ | 캔들 형태 복합 지표 |
| 2 | f3 | lnHO·lnCH / body | 상방 꼬리 비율 |
| 3 | f4 | lnLO·lnCL / body | 하방 꼬리 비율 |
| 4 | log_volume | ln(V_t / V_{t-1}) | 거래량 로그 수익률 |
| 5 | funding_rate | 원본값 | 바이낸스 펀딩비 |
| 6 | basis | 원본값 | 선물-현물 괴리율 |
| 7 | log_ls_ratio | ln(LS_t / LS_{t-1}) | 롱숏비 로그 수익률 |
| 8 | log_oi | ln(OI_t / OI_{t-1}) | 미결제약정 로그 수익률 |

피처 f2, f3, f4는 `[-10, 10]`으로 클리핑. 입력 시퀀스는 z-score 정규화(피처별 mean/std) 적용.

### 타겟 피처 (5개)

| 인덱스 | 이름 | 정의 |
|---|---|---|
| 0 | lnCO | ln(Close / Open) |
| 1 | lnHO | ln(High / Open) |
| 2 | lnLO | ln(Low / Open) |
| 3 | lnCH | ln(Close / High) |
| 4 | lnCL | ln(Close / Low) |

5개의 log-return으로 각 캔들의 OHLC 형태를 완전히 복원할 수 있다.

---

## 프로젝트 구조

```
quarter_v1/
├── config.py              # 전역 상수 (피처, 시퀀스 길이, API 설정, 학습 배치 크기)
├── train.py               # 학습 스크립트 (EMA, 스케줄러, 체크포인트, 시각화)
├── inference.py            # 추론 스크립트 (미구현)
├── utils.py                # 유틸리티 (미구현)
├── requirements.txt        # 의존성
├── core/
│   ├── __init__.py
│   ├── heo.py             # HeLU, HeoGate, Heoptimizer, HeoLoss
│   ├── block.py           # EmbeddingBlock, AttentionBlock, FFNBlock, QuarterBlock, RoPE
│   └── net.py             # QuarterNet (최종 모델)
├── data/
│   ├── __init__.py
│   ├── apicalling.py      # BinanceFetcher (Futures API 수집)
│   └── dataset.py         # 피처 엔지니어링, HDF5 생성, QuarterDataset
├── dataset/                # HDF5 데이터셋 (git 추적 제외)
├── checkpoints/            # 모델 체크포인트 (git 추적 제외)
└── outputs/                # 시각화 이미지 (git 추적 제외)
```

---

## 데이터 파이프라인

```
Binance Futures API
  ├─ klines        (OHLCV, 15m)
  ├─ funding_rate
  ├─ basis
  ├─ long_short_ratio
  └─ open_interest
        │
        ▼
   시간축 정렬 (ffill + dropna)
        │
        ▼
   피처 엔지니어링
   ├─ 캔들 로그 수익률 (lnCO, lnHO, lnLO, lnCH, lnCL)
   ├─ 복합 형태 지표 (f2, f3, f4)
   ├─ 거래량/롱숏비/미결제약정 로그 수익률
   └─ z-score 정규화
        │
        ▼
   슬라이딩 윈도우
   ├─ input:  (n_samples, 9600, 9)
   └─ target: (n_samples, 96, 5)
        │
        ▼
   HDF5 저장 (gzip 압축, chunk=(1, seq_len, features))
```

### 데이터 기간

| 구분 | 시작 | 종료 |
|---|---|---|
| 학습 | 2020-01-01 | 2025-06-30 |
| 평가 | 2025-07-01 | 2025-12-31 |

데이터셋이 없으면 학습 시작 시 자동으로 Binance API에서 수집하여 빌드한다.

---

## 학습

### 하이퍼파라미터

| 항목 | 값 |
|---|---|
| 배치 크기 | 1 (`config.py`에서 조절) |
| 옵티마이저 | Heoptimizer (AdamW 기반) |
| 학습률 | 1e-4 |
| Weight Decay | 1e-4 (HeLU/HeoGate 제외) |
| EMA 계수 | 0.999 |
| 스케줄러 | 1 에폭 Linear Warmup → ExponentialLR (γ=0.999998) |
| 손실 함수 | HeoLoss |
| 평가 주기 | 1,024 스텝 |
| 로그 주기 | 64 스텝 |

### 사용법

```bash
# 이어서 학습 (기본값, 체크포인트 없으면 처음부터)
python train.py

# 명시적 이어서 학습
python train.py --resume

# 처음부터 새로 학습 (기존 체크포인트 유지)
python train.py --restart
```

### 학습 흐름

- **무한 루프**: 에폭 제한 없이 수동 중단(Ctrl+C)까지 학습
- **EMA**: 매 스텝 exponential moving average 갱신. 평가와 시각화는 EMA 파라미터로 수행
- **스케줄러**: 첫 1 에폭 동안 lr을 1e-7 배율에서 1.0 배율로 선형 증가 후, 매 스텝 γ=0.999998로 지수 감소
- **체크포인트**: 1,024 스텝마다 `checkpoints/last.pt` 저장. eval loss 갱신 시 `checkpoints/best.pt` 추가 저장
- **시각화**: 평가 시마다 정답과 예측을 OHLC 캔들차트로 복원하여 `outputs/step_XXXXXX.png`에 저장

---

## 의존성

### 시스템 패키지

```bash
apt-get update && apt-get install -y fonts-nanum
rm -rf $(python -c "import matplotlib; print(matplotlib.get_cachedir())")
```

시각화에 한글 폰트(NanumGothic)를 사용한다. 설치 후 matplotlib 폰트 캐시를 삭제해야 인식된다.

### Python 패키지

```
requests>=2.32
pandas>=2.2
numpy>=2.2
h5py>=3.11
torch>=2.9
matplotlib>=3.9
```

```bash
git clone https://github.com/Tom-Heo/quarter_v1.git
cd quarter_v1
pip install -r requirements.txt
apt-get update && apt-get install -y fonts-nanum
rm -rf $(python -c "import matplotlib; print(matplotlib.get_cachedir())")
python train.py --resume
```

FEATURES = [
    "f1",
    "f2",
    "f3",
    "f4",
    "lnHO",
    "lnLO",
    "lnCH",
    "lnCL",
    "lnHL",
    "log_volume",
    "log_trades",
    "taker_buy_ratio",
    "log_oi",
    "long_short_ratio",
    "funding_rate",
    "basis",
    "sin_hour",
    "cos_hour",
    "sin_dow",
    "cos_dow",
]

NUM_FEATURES = len(FEATURES)

SEQ_LEN = 9600
STRIDE = 1

EPSILON = 1e-8
CLIP_BOUND = 10.0

TARGET_LEN = 96
TARGET_FEATURES = ["lnCO", "lnHO", "lnLO", "lnCH", "lnCL"]
NUM_TARGET_FEATURES = len(TARGET_FEATURES)
LOSS_WEIGHTS = [16.0, 1.0, 1.0, 1.0, 1.0]

BINANCE_SYMBOL = "BTCUSDT"
BINANCE_PAIR = "BTCUSDT"
BINANCE_INTERVAL = "15m"
BINANCE_KLINE_LIMIT = 1500
BINANCE_FUNDING_LIMIT = 1000
BINANCE_DATA_LIMIT = 500
BINANCE_SLEEP = 0.4
BINANCE_DATA_SLEEP = 1.2

DATASET_DIR = "dataset"
CHECKPOINT_DIR = "checkpoints"

TRAIN_DATASET_START = "2020-01-01"
TRAIN_DATASET_END = "2025-06-30"
EVAL_DATASET_START = "2025-07-01"
EVAL_DATASET_END = "2025-12-31"

BATCH_SIZE = 1

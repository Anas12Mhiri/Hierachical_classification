"""Central configuration. Override the data/output dirs via CLI flags or env vars."""
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get(
    "ARAGENRE_DATA_DIR",
    "/kaggle/input/datasets/anasmhiri/aragenre/public_data_train_dev",
))
OUT_DIR = Path(os.environ.get("ARAGENRE_OUT_DIR", "baseline_outputs"))

DEV_TEXT_FILE_NAME = "dev.json"
DEV_GOLD_FILE_NAME = "dev_gold.json"
DEV_DEFINITIONS_FILE_NAME = "dev_genre_definitions.json"

DETAILED_OUT_NAME = "baseline_zeroshot_dev_predictions_detailed.json"
SUBMISSION_OUT_NAME = "baseline_zeroshot_dev_submission.json"

# Models
EMBEDDING_MODEL = "sentence-transformers/sentence-t5-xxl"
QWEN_MODEL = "Qwen/Qwen3-4B"
# Alternative (GPU) translator, not used by default:
TRANSLATION_MODEL = "facebook/nllb-200-1.3B"

# Devices
QWEN_DEVICE = "cuda:1"
EMBED_DEVICE = "cuda:0"

# Generation / batching
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.0
TRANSLATION_BATCH_SIZE = 8
EMBEDDING_BATCH_SIZE = 32

# Ensemble weights: raw text similarity vs Qwen explanation similarity
WEIGHT_TEXT = 0.4
WEIGHT_PRED = 0.6

RUN_LIMIT = None            # e.g. 10 to smoke-test
SLEEP_BETWEEN_CALLS = 0.0

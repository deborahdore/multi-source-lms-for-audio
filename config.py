import os
from pathlib import Path

# ======================== PATH ======================== #
ABSOLUTE_PATH = os.path.abspath(".")
DATASET_DIR = os.path.join(ABSOLUTE_PATH, "slakh2100")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
VAL_DIR = os.path.join(DATASET_DIR, "validation")
LOGGING_DIR = os.path.join(ABSOLUTE_PATH, "logging")
OUTPUT_DIR = os.path.join(ABSOLUTE_PATH, "output_samples")

# ======================== MODEL CONFIG ======================== #
NUM_HIDDEN = 128
NUM_RESIDUAL_HIDDEN = 32
NUM_RESIDUAL_LAYER = 2
EMBEDDING_DIM = 64
NUM_EMBEDDINGS = 512
COMMITMENT_COST = 0.25

# ======================== RUN CONFIG ======================== #
LEARNING_RATE = 2e-4
LIMIT_TRAIN_BATCHES = 100
MAX_EPOCHS = 100
BATCH_SIZE = 128
PROFILER = "simple"

# ======================== LOGGING CONFIG ======================== #
WANDB = True
WANDB_OFFLINE = False
WANDB_PROJECT_NAME = "VQVAE_experiments"

# ======================== CREATING PATHS ======================== #
assert Path(TRAIN_DIR).exists()
assert Path(VAL_DIR).exists()
assert Path(TEST_DIR).exists()

Path(LOGGING_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
if WANDB:
	Path(LOGGING_DIR + "/wandb").mkdir(parents=True, exist_ok=True)

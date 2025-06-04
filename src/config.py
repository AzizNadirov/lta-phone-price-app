from dataclasses import dataclass


# Phone price segments for ML training
ML_PHONE_SEGMENTS = {
    "budget": (0, 300),
    "mid_range": (300, 700),
    "premium": (700, float("inf"))}

# Phone specs-price dataset path for ML
ML_PHONE_DATASET_PATH = "src/data/phone-price.parquet"

# feature columns list
feature_list = ("RAM_GB", "ROM_GB", "RAM_MB", "ROM_MB", "NFC", "camera_mp", "CPU", "brand", "OS")

# Parsing LLM confs
PARSING_LLM_ATTEMPTS = 1 
PARSING_LLM_WAIT = 1

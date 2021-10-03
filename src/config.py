from pathlib import Path

class Config:
    """
    This class shall be used for getting the configuration for the scripts.
    """
    RANDOM_STATE = 42
    ASSETS_PATH = Path("./assets")
    Bundle = "secure-connect-rental-bike-sharing.zip"
    ORIGINAL_DATASET_FILE_PATH = ASSETS_PATH / "original_dataset"
    PROCESSED_DATASET_PATH = ASSETS_PATH / "processed_data" 
    SPLIT_DATASET_PATH = ASSETS_PATH / "split_data" 
    PICKLE_FILES_PATH = ASSETS_PATH / "saved_models" 
    DAY_CAT_COLS = ['season', 'mnth', 'weekday', 'weathersit', 'workingday', 'holiday']
    HOUR_CAT_COLS = ['season', 'mnth', 'weekday', 'hr', 'weathersit','workingday','holiday']
    COLS_TO_DROP = ['instant', 'atemp', 'cnt', 'dteday', 'yr']
    DAY_METRICS_FILE_PATH = ASSETS_PATH / "day_metrics.json"
    HOUR_METRICS_FILE_PATH = ASSETS_PATH / "hour_metrics.json"
    USER_INPUTS_FILE_PATH = ASSETS_PATH / "user_input" 

    # FEATURES_PATH = ASSETS_PATH / "features"
    # 
    # LOGS_PATH = ASSETS_PATH / "logs"
    # TUNED_HYPERPARAMS_FILE_PATH = ASSETS_PATH / "best_params.json"
    # NOTEBOOKS_PATH = ASSETS_PATH / "notebooks"


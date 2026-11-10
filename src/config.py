from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Raw Files
HOUSE_DATA_PATH = RAW_DATA_DIR / "HouseTS.csv"
CITIES_DATA_PATH = RAW_DATA_DIR / "uscities.csv"

# Temporal Split Cutoffs
CUTOFF_DATE_EVAL = "2020-01-01"      # Validation starts (Covid period)
CUTOFF_DATE_HOLDOUT = "2022-01-01"   # Holdout test starts (Inflation period)

# Outlier Thresholds
MAX_LIST_PRICE = 19_000_000
MAX_PRICE_RATIO = 10.0

# Target Column
TARGET_COL = "price"

# Demographic columns with missing values (coded as zeros)
DEMOGRAPHIC_COLS = [
    "Total Population",
    "Median Age",
    "Per Capita Income",
    "Median Rent",
    "Median Home Value",
    "Total Housing Units",
]
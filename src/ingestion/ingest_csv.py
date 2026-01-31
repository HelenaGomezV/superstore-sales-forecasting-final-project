import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DEFAULT_RAW_PATH = BASE_DIR / "data" / "raw" / "Warehouse_and_Retail_Sales.csv"
DEFAULT_PROCESSED_DIR = BASE_DIR / "data" / "processed"

def ingest_csv(raw_path: Path = DEFAULT_RAW_PATH,
               processed_dir: Path = DEFAULT_PROCESSED_DIR):

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    df = pd.read_csv(raw_path)

    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_dir / "processed.csv", index=False)


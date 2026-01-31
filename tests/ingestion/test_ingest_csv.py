import pytest
from pathlib import Path
from src.ingestion.ingest_csv import ingest_csv

def test_ingestion_fails_if_raw_file_missing(tmp_path):
    fake_raw_path = tmp_path / "missing.csv"
    processed_dir = tmp_path / "processed"

    with pytest.raises(FileNotFoundError):
        ingest_csv(fake_raw_path, processed_dir)

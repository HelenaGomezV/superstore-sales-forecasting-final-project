import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"

DATA_PATH = DATA_DIR / "Sample - Superstore.csv"
KNOWLEDGE_BASE_PATH = SRC_DIR / "knowledge_base.txt"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

APP_TITLE = "Superstore Sales Analyst"
APP_ICON = " "
MAX_CHAT_HISTORY = 10
CHAT_TEMPERATURE = 0.3
MAX_TOKENS = 1500

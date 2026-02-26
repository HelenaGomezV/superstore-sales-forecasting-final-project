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
APP_ICON = "\U0001F4CA"
MAX_CHAT_HISTORY = 10
CHAT_TEMPERATURE = 0.3
MAX_TOKENS = 1500

CHART_COLORS = ["#1DB954", "#1ED760", "#A0E77D", "#509BF5", "#F573A0",
                "#E8125C", "#FFA42B", "#F5E6C8", "#B49BC8", "#CDF564"]

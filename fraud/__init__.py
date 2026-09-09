"""Credit-card fraud experiment addressing the manuscript review."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DATA_PATH = PROJECT_ROOT / "creditcard.csv"
OUTPUT_DIR = PACKAGE_DIR / "output"

import os
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "../../../data")
HTML_DIR = os.path.join(DATA_DIR, "html")
GOLD_DATA_DIR = os.path.join(DATA_DIR, "processed")
DEFAULT_GOLD_DATA_DIR = os.path.join(GOLD_DATA_DIR, "all-train")


"""Top-level entry point to run the Payment Routing REST API."""

import sys
from pathlib import Path

# Ensure src/ is on sys.path
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import uvicorn

if __name__ == "__main__":
    print("Starting Cross-Border Payment Routing REST API on http://127.0.0.1:8000 ...")
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)

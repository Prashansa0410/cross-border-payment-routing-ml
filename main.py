"""Top-level entry point to run the Payment Routing REST API."""

import os
import sys
from pathlib import Path

# Ensure src/ is on sys.path
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import uvicorn

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Cross-Border Payment Routing REST API on http://{host}:{port} ...")
    uvicorn.run("api.app:app", host=host, port=port, reload=False)


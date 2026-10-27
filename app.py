"""Convenience entry point: `python app.py` starts the dashboard on http://127.0.0.1:5000.

Equivalent to `sar-flood serve`. Configure with the environment variables documented in
src/sar_flood/web/app.py (SAR_FLOOD_RUN_DIR, SECRET_KEY, MAX_UPLOAD_MB, ...).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # works without `pip install -e .`

from sar_flood.web.app import create_app  # noqa: E402

app = create_app()

if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5000")))

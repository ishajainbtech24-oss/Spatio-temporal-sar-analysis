"""Regenerate the bundled synthetic demo chips in samples/synthetic/.

They follow the Sen1Floods11 S1Hand/LabelHand format so the dashboard can be demonstrated
without downloading the dataset. They are synthetic and say nothing about real skill.
"""

import shutil
import tempfile
from pathlib import Path

from sar_flood.data.sen1floods11 import LABEL_SUBDIR, S1_SUBDIR
from sar_flood.data.synthetic import write_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "samples" / "synthetic"

with tempfile.TemporaryDirectory() as tmp:
    write_dataset(tmp, n_per_event=2, size=256, seed=123)
    OUT.mkdir(parents=True, exist_ok=True)
    for chip in ["Foxtrot_1000", "Foxtrot_1001"]:  # Foxtrot = synthetic "test" event
        shutil.copy(Path(tmp) / S1_SUBDIR / f"{chip}_S1Hand.tif", OUT / f"Synthetic_{chip}_S1Hand.tif")
        shutil.copy(Path(tmp) / LABEL_SUBDIR / f"{chip}_LabelHand.tif", OUT / f"Synthetic_{chip}_LabelHand.tif")
print(f"wrote demo chips to {OUT}")

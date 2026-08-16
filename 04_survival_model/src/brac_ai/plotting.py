from __future__ import annotations

import os
import tempfile
from pathlib import Path


if "MPLCONFIGDIR" not in os.environ:
    config_dir = Path(tempfile.gettempdir()) / "brac_ai_matplotlib"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(config_dir)

import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt  # noqa: E402

__all__ = ["plt"]


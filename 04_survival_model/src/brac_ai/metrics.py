from __future__ import annotations
import numpy as np


def concordance_index(time, risk, event) -> float:
    """Harrell C-index; larger risk means earlier event."""
    time = np.asarray(time, dtype=float)
    risk = np.asarray(risk, dtype=float)
    event = np.asarray(event, dtype=int)
    concordant = 0.0
    comparable = 0.0
    for i in range(len(time)):
        if event[i] != 1:
            continue
        for j in range(len(time)):
            if time[j] <= time[i]:
                continue
            comparable += 1.0
            if risk[i] > risk[j]:
                concordant += 1.0
            elif risk[i] == risk[j]:
                concordant += 0.5
    return float(concordant / comparable) if comparable else float("nan")

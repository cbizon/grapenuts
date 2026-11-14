"""Where does the layer stop depending on stiffness?

The stalling condition of the layer pilot (same protocol and seeds) at
k = 1e8 with 20 kicks per contact time, and at k = 1e7 with 50.  Writes
results/stiff_check.json.

    uv run python scripts/stiff_check.py
"""

import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer_pilot  # noqa: E402

layer_pilot.METHODS.update({
    "tc k=1e8": dict(contact_mode="tc", stiffness=1e8),
    "tc k=1e7, 50 kicks": dict(contact_mode="tc", stiffness=1e7, steps_per_contact=50.0),
})
JOBS = [("stalls hard", method, seed)
        for method in ("tc k=1e8", "tc k=1e7, 50 kicks")
        for seed in layer_pilot.SEEDS]


def job(args):
    condition, method, seed = args
    return dict(condition=condition, method=method, seed=seed,
                **layer_pilot.run(condition, method, seed))


if __name__ == "__main__":
    with Pool(len(JOBS)) as pool:
        rows = pool.map(job, JOBS)
    path = Path(__file__).resolve().parent.parent / "results" / "stiff_check.json"
    path.write_text(json.dumps(dict(runs=rows), indent=1) + "\n")
    print("wrote", path)

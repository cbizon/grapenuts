"""Hard against hybrid across stiffness at the flat, hard-landing condition
found by flat_scan.py: Gamma 5, f* 0.6, 10 layers, L = 100.

Same measurements as the layer pilot, three seeds, 60 cycles to settle and
100 measured.  Writes results/flat_compare.json.

    uv run python scripts/flat_compare.py
"""

import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer_pilot  # noqa: E402

layer_pilot.SETTLE = 60
layer_pilot.CONDITIONS["flat"] = dict(gamma=5.0, f_star=0.6, depth=10)
layer_pilot.METHODS = {"hard": dict(contact_mode="never")}
for stiffness in ("1e5", "1e6", "1e7", "1e8", "1e9"):
    layer_pilot.METHODS["tc k=" + stiffness] = dict(contact_mode="tc",
                                                    stiffness=float(stiffness))
JOBS = [("flat", method, seed) for method in layer_pilot.METHODS
        for seed in layer_pilot.SEEDS]


def job(args):
    condition, method, seed = args
    return dict(condition=condition, method=method, seed=seed,
                **layer_pilot.run(condition, method, seed))


if __name__ == "__main__":
    with Pool(9) as pool:
        rows = pool.map(job, JOBS, chunksize=1)
    path = Path(__file__).resolve().parent.parent / "results" / "flat_compare.json"
    path.write_text(json.dumps(dict(runs=rows), indent=1) + "\n")
    print("wrote", path)

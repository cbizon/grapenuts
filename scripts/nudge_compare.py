"""The nudge engine (hard spheres plus sequel's position nudge onto the
plate) at the flat condition, as a baseline independent of the springs.

Gamma 5, f* 0.6, 10 layers, L = 100, three seeds, from the default start and
from the half-push start, 60 cycles to settle and 100 measured.  Writes
results/nudge_compare.json.

    uv run python scripts/nudge_compare.py
"""

import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer_pilot  # noqa: E402

layer_pilot.SETTLE = 60
layer_pilot.CONDITIONS["flat"] = dict(gamma=5.0, f_star=0.6, depth=10)
layer_pilot.METHODS = {"nudge": dict(contact_mode="never", plate_nudge=True)}
JOBS = [("nudge", seed, push) for push in (0.0, 0.5) for seed in layer_pilot.SEEDS]


def job(args):
    method, seed, push = args
    return dict(condition="flat", method=method, seed=seed, push=push,
                **layer_pilot.run("flat", method, seed, push=push))


if __name__ == "__main__":
    path = Path(__file__).resolve().parent.parent / "results" / "nudge_compare.json"
    rows = []
    with Pool(len(JOBS)) as pool:
        for row in pool.imap_unordered(job, JOBS):
            rows.append(row)
            path.write_text(json.dumps(dict(runs=rows), indent=1) + "\n")
            print(row["method"], "push", row["push"], "seed", row["seed"],
                  "finished" if row["finished"] else row["failure"],
                  "cycles %.2f" % row["cycles"], flush=True)

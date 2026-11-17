"""Can the hard engine run the flat condition from a gentler start?

Gamma 5, f* 0.6, 10 layers, L = 100, with every sphere given an initial
upward speed of a fraction of the plate's peak speed (the plate starts at
mid-height moving up at peak speed, so the default start is hit at full
speed).  Hard engine at four fractions, and the hybrid at k = 1e7 from the
half-push start as a control; three seeds, 60 cycles to settle and 100
measured.  Writes results/push_start.json.

    uv run python scripts/push_start.py
"""

import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer_pilot  # noqa: E402

layer_pilot.SETTLE = 60
layer_pilot.CONDITIONS["flat"] = dict(gamma=5.0, f_star=0.6, depth=10)
layer_pilot.METHODS = {"hard": dict(contact_mode="never"),
                       "tc k=1e7": dict(contact_mode="tc", stiffness=1e7)}
JOBS = ([("hard", seed, push) for push in (0.25, 0.5, 0.75, 1.0)
         for seed in layer_pilot.SEEDS]
        + [("tc k=1e7", seed, 0.5) for seed in layer_pilot.SEEDS])


def job(args):
    method, seed, push = args
    return dict(condition="flat", method=method, seed=seed, push=push,
                **layer_pilot.run("flat", method, seed, push=push))


if __name__ == "__main__":
    path = Path(__file__).resolve().parent.parent / "results" / "push_start.json"
    rows = []
    with Pool(12) as pool:
        for row in pool.imap_unordered(job, JOBS):
            rows.append(row)
            path.write_text(json.dumps(dict(runs=rows), indent=1) + "\n")
            print(row["method"], "push", row["push"], "seed", row["seed"],
                  "finished" if row["finished"] else row["failure"],
                  "cycles %.2f" % row["cycles"], flush=True)

"""Does the layer's dependence on stiffness come from the time step?

Reruns the stalling condition of the layer pilot (same protocol and seeds)
with 50 kicks per contact time instead of 20, at k = 1e5 and 1e6.  Writes
results/steps_check.json.

    uv run python scripts/steps_check.py
"""

import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer_pilot  # noqa: E402

layer_pilot.METHODS.update({
    "tc k=1e5, 50 kicks": dict(contact_mode="tc", stiffness=1e5, steps_per_contact=50.0),
    "tc k=1e6, 50 kicks": dict(contact_mode="tc", stiffness=1e6, steps_per_contact=50.0),
})
JOBS = [("stalls hard", method, seed)
        for method in ("tc k=1e5, 50 kicks", "tc k=1e6, 50 kicks")
        for seed in layer_pilot.SEEDS]


def job(args):
    condition, method, seed = args
    return dict(condition=condition, method=method, seed=seed,
                **layer_pilot.run(condition, method, seed))


if __name__ == "__main__":
    with Pool(len(JOBS)) as pool:
        rows = pool.map(job, JOBS)
    path = Path(__file__).resolve().parent.parent / "results" / "steps_check.json"
    path.write_text(json.dumps(dict(runs=rows), indent=1) + "\n")
    print("wrote", path)

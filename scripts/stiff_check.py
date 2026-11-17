"""Where does the layer stop depending on stiffness?

Runs the stalling condition of the layer pilot (same protocol and seeds) with
the tc trigger at the given stiffnesses and kicks per contact time, and
writes the runs to the given file under results/.

    uv run python scripts/stiff_check.py stiff_check.json 1e8:20 1e7:50
    uv run python scripts/stiff_check.py stiff_check_1e9.json 1e9:20
"""

import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer_pilot  # noqa: E402

JOBS = []
for spec in sys.argv[2:]:
    stiffness, kicks = spec.split(":")
    name = "tc k=%s" % stiffness + ("" if kicks == "20" else ", %s kicks" % kicks)
    layer_pilot.METHODS[name] = dict(contact_mode="tc", stiffness=float(stiffness),
                                     steps_per_contact=float(kicks))
    JOBS += [("stalls hard", name, seed) for seed in layer_pilot.SEEDS]


def job(args):
    condition, method, seed = args
    return dict(condition=condition, method=method, seed=seed,
                **layer_pilot.run(condition, method, seed))


if __name__ == "__main__":
    with Pool(len(JOBS)) as pool:
        rows = pool.map(job, JOBS)
    path = Path(__file__).resolve().parent.parent / "results" / sys.argv[1]
    path.write_text(json.dumps(dict(runs=rows), indent=1) + "\n")
    print("wrote", path)

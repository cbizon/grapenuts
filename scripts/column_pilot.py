"""Pilot: overlap, energy error and cost of the hybrid column against spring
stiffness, time step and trigger.

A column of 10 spheres driven as a layer 10 deep at Gamma = 5, f* = 0.44,
three seeds (diameters), 200 cycles each.  Writes results/column_pilot.json.

    uv run python scripts/column_pilot.py
"""

import json
import math
import time
from pathlib import Path

import numpy as np

from granular import column as C

N, GAMMA, F_STAR, CYCLES, SEEDS = 10, 5.0, 0.44, 200, (1, 2, 3)
MAX_TOUCHES = 20_000_000
TRIGGERS = {
    "always": dict(mode=C.ALWAYS),
    "tc": dict(mode=C.TC),
    "tc, springs below 0.1": dict(mode=C.TC, soft_speed_limit=0.1),
    "tc, springs below 0.01": dict(mode=C.TC, soft_speed_limit=0.01),
}


def run(stiffness, steps, trigger, seed):
    column, period = C.driven_column(N, GAMMA, F_STAR, stiffness, seed=seed,
                                     steps_per_contact=steps, **TRIGGERS[trigger])
    energy0 = column.energy()
    start = time.process_time()
    finished = column.run_to(CYCLES * period, max_touches=MAX_TOUCHES)
    cpu = time.process_time() - start
    cycles = column.t / period
    s = column.stats
    return dict(
        finished=bool(finished), cycles=cycles,
        max_overlap=float(max(s[C.S_MAX_PAIR_OVERLAP], s[C.S_MAX_PLATE_OVERLAP])),
        max_spring_speed=float(s[C.S_MAX_SPRING_SPEED]),
        kicks_per_sphere_cycle=float(s[C.S_KICK_FORCES] / N / cycles),
        hard_per_sphere_cycle=float((s[C.S_HARD_PAIR] + s[C.S_HARD_PLATE]) / N / cycles),
        energy_residual=float((column.energy() - energy0 - column.accounted())
                              / np.abs(column.acc[:3]).sum()),
        mean_height=float(column.acc[C.A_ZCM_INT] / column.t),
        cpu_seconds=cpu)


def summarize(rows):
    keys = [k for k in rows[0] if k != "finished"]
    out = dict(finished=sum(r["finished"] for r in rows), seeds=len(rows))
    for key in keys:
        values = [r[key] for r in rows]
        out[key] = dict(mean=float(np.mean(values)), min=float(np.min(values)),
                        max=float(np.max(values)))
    return out


def main():
    # warm up the compiled kernel so timings exclude compilation
    C.driven_column(2, GAMMA, F_STAR, 6e4, C.ALWAYS)[0].run_to(1.0)
    results = dict(conditions=dict(n=N, gamma=GAMMA, f_star=F_STAR,
                                   cycles=CYCLES, seeds=list(SEEDS),
                                   max_touches=MAX_TOUCHES),
                   stiffness_sweep=[], step_sweep=[], hard=[])
    for trigger in TRIGGERS:
        for stiffness in (6e4, 6e5, 6e6, 6e7):
            rows = [run(stiffness, 20.0, trigger, seed) for seed in SEEDS]
            results["stiffness_sweep"].append(dict(
                trigger=trigger, stiffness=stiffness, steps_per_contact=20.0,
                contact_time=math.pi * math.sqrt(0.5 / stiffness), **summarize(rows)))
    for steps in (10.0, 20.0, 40.0, 80.0):
        rows = [run(6e5, steps, "tc", seed) for seed in SEEDS]
        results["step_sweep"].append(dict(trigger="tc", stiffness=6e5,
                                          steps_per_contact=steps, **summarize(rows)))
    for seed in SEEDS:
        column, period = C.driven_column(N, GAMMA, F_STAR, 6e5, C.NEVER, seed=seed)
        finished = column.run_to(CYCLES * period, max_touches=MAX_TOUCHES)
        results["hard"].append(dict(seed=seed, finished=bool(finished),
                                    cycles=column.t / period))
    path = Path(__file__).resolve().parent.parent / "results" / "column_pilot.json"
    path.write_text(json.dumps(results, indent=1) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()

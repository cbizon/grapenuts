"""Pilot: a 2D layer with hard contacts against the hybrid, at one condition
the hard engine runs and one where it stalls.

L = 100, three seeds, 20 cycles to settle then 100 measured, sampled at 20
phases per cycle.  Writes results/layer_pilot.json.

    uv run python scripts/layer_pilot.py
"""

import json
import time
from pathlib import Path

import numpy as np

from granular import edmd as K
from granular.sim import Simulation

L, SEEDS, SETTLE, MEASURE, PHASES = 100.0, (1, 2, 3), 20, 100, 20
MAX_EVENTS_PER_STEP = 300_000_000      # a phase step that needs more has stalled
CONDITIONS = {
    "runs hard": dict(gamma=3.5, f_star=0.5, depth=5),
    "stalls hard": dict(gamma=5.0, f_star=0.3, depth=10),
}
METHODS = {
    "hard": dict(contact_mode="never"),
    "tc k=1e5": dict(contact_mode="tc", stiffness=1e5),
    "tc k=1e6": dict(contact_mode="tc", stiffness=1e6),
    "tc k=1e7": dict(contact_mode="tc", stiffness=1e7),
    "tc k=1e6, springs below 0.1": dict(contact_mode="tc", stiffness=1e6,
                                        soft_speed_limit=0.1),
    "always k=1e6": dict(contact_mode="always", stiffness=1e6),
}


def run(condition, method, seed):
    c = CONDITIONS[condition]
    n = int(c["depth"] * L)
    sim = Simulation(n, L, c["gamma"], c["f_star"], float(c["depth"]), seed,
                     dim=2, **METHODS[method])
    start = time.process_time()
    heights, kinetic = [], []
    energy0 = counts0 = None
    for cycle in range(SETTLE + MEASURE):
        if cycle == SETTLE:
            energy0 = sim.layer_energy()[0]
            counts0 = (sim.cnt.copy(), sim.en.copy())
        for step in range(1, PHASES + 1):
            if sim.run_to(sim.T * step / PHASES, max_events=MAX_EVENTS_PER_STEP):
                return dict(finished=False, cycles=cycle + (step - 1) / PHASES,
                            cpu_seconds=time.process_time() - start)
            if cycle >= SETTLE:
                d = sim.diagnostics()
                heights.append(d["zcm"])
                kinetic.append((d["ke"] + d["krot"]) / n)
        sim.end_cycle()
    cnt = sim.cnt - counts0[0]
    en = sim.en - counts0[1]
    terms = [en[K.EN_PLATE_WORK], en[K.EN_PLATE_DISS], en[K.EN_PP_DISS],
             en[K.EN_WALL_DISS], en[K.EN_SOFT_DISS]]
    summary = sim.soft_summary()
    per = n * MEASURE
    return dict(
        finished=True, cycles=SETTLE + MEASURE,
        cpu_seconds=time.process_time() - start,
        mean_height=float(np.mean(heights)),
        mean_kinetic_per_sphere=float(np.mean(kinetic)),
        energy_residual=float((sim.layer_energy()[0] - energy0 - sum(terms))
                              / sum(abs(x) for x in terms)),
        hard_collisions_per_sphere_cycle=float((cnt[K.C_PP] * 2 + cnt[K.C_PLATE]) / per),
        kick_forces_per_sphere_cycle=float(cnt[K.C_KICKF] * 2 / per),
        springs_made_per_sphere_cycle=float(cnt[K.C_SPRINGS] * 2 / per),
        max_overlap=max(summary["max_pair_overlap"], summary["max_plate_overlap"]),
        max_spring_speed=summary["max_spring_speed"])


def main():
    Simulation(20, 10.0, 3.5, 0.5, 2.0, 1, dim=2, contact_mode="tc",
               stiffness=1e5).run_to(1.0)       # compile before timing
    path = Path(__file__).resolve().parent.parent / "results" / "layer_pilot.json"
    results = dict(setup=dict(L=L, seeds=list(SEEDS), settle=SETTLE,
                              measure=MEASURE, phases=PHASES,
                              max_events_per_step=MAX_EVENTS_PER_STEP,
                              conditions=CONDITIONS,
                              methods={k: {a: b for a, b in v.items()}
                                       for k, v in METHODS.items()}),
                   runs=[])
    for condition in CONDITIONS:
        for method in METHODS:
            for seed in SEEDS:
                row = dict(condition=condition, method=method, seed=seed,
                           **run(condition, method, seed))
                results["runs"].append(row)
                path.write_text(json.dumps(results, indent=1) + "\n")
                print(condition, "|", method, "| seed", seed, "|",
                      "finished" if row["finished"] else "STALLED",
                      "| cycles %.2f | cpu %.1f s" % (row["cycles"], row["cpu_seconds"]),
                      flush=True)


if __name__ == "__main__":
    main()

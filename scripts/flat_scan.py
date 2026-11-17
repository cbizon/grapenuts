"""Scan for a regime where the layer stays flat and lands hard.

2D, 10 layers, hybrid (tc, k = 1e6, 20 kicks per contact time), seed 1,
60 cycles to settle then 40 measured.  Per run: how flat the layer is (the
spread along the box of the column centre-of-mass height and of the column
occupancy, in columns 2 D wide), and how hard it lands (largest plate force
over the layer's weight, from the plate impulse in 1/200ths of a cycle, and
the largest share of spheres holding a spring at once).  Writes
results/flat_scan.json, with snapshots of the last two cycles.

    uv run python scripts/flat_scan.py
"""

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from granular import edmd as K
from granular.sim import Simulation

DEPTH, SETTLE, MEASURE, SUB = 10, 60, 40, 200
CASES = ([dict(gamma=g, f_star=f, L=100.0) for g in (4.5, 5.0, 5.5)
          for f in (0.44, 0.6, 0.8)]
         + [dict(gamma=5.0, f_star=0.3, L=100.0),
            dict(gamma=5.0, f_star=0.3, L=20.0),
            dict(gamma=5.0, f_star=0.44, L=20.0)])


def run(case):
    L = case["L"]
    n = int(DEPTH * L)
    sim = Simulation(n, L, case["gamma"], case["f_star"], float(DEPTH), 1, dim=2,
                     contact_mode="tc", stiffness=1e6)
    weight = float(sim.mass.sum())
    for _ in range(SETTLE):
        if sim.run_to(sim.T, max_events=300_000_000):
            return dict(case, finished=False)
        sim.end_cycle()
    nb = int(L / 2)
    rough, occupancy, modes, forces, springs, impulses, snaps = [], [], [], [], [], [], []
    heights, kinetic = [], []
    for cycle in range(MEASURE):
        cycle_impulse = 0.0
        for step in range(1, SUB + 1):
            before = sim.en[K.EN_PLATE_JZ]
            if sim.run_to(sim.T * step / SUB, max_events=300_000_000):
                return dict(case, finished=False)
            impulse = sim.en[K.EN_PLATE_JZ] - before
            cycle_impulse += impulse
            forces.append(impulse / (sim.T / SUB) / weight)
            springs.append(sim.si[K.SI_NC] / n)
            if step % 10 == 0:
                r, v, w = sim.snapshot()
                column = np.clip((r[:, 0] / L * nb).astype(int), 0, nb - 1)
                count = np.bincount(column, minlength=nb)
                centre = np.bincount(column, weights=r[:, 2], minlength=nb) / np.maximum(count, 1)
                centre = np.where(count > 0, centre, np.nan)
                rough.append(float(np.nanstd(centre)))
                occupancy.append(float(np.std(count) / np.mean(count)))
                modes.append(np.abs(np.fft.rfft(np.nan_to_num(centre - np.nanmean(centre)))) * 2 / nb)
                d = sim.diagnostics()
                heights.append(d["zcm"])
                kinetic.append((d["ke"] + d["krot"]) / n)
                if cycle >= MEASURE - 2 and step % 50 == 0:
                    snaps.append(dict(cycle=cycle, phase=step / SUB,
                                      zp=sim.plate(sim.t_local)[0],
                                      x=np.round(r[:, 0], 2).tolist(),
                                      z=np.round(r[:, 2], 2).tolist()))
        impulses.append(cycle_impulse)
        sim.end_cycle()
    amplitude = np.mean(modes, axis=0)
    top = int(np.argmax(amplitude[1:])) + 1
    impulses = np.array(impulses)
    even, odd = impulses[0::2].mean(), impulses[1::2].mean()
    forces = np.array(forces).reshape(MEASURE, SUB)
    summary = sim.soft_summary()
    return dict(case, finished=True, period=sim.T, amplitude=sim.A,
                plate_speed=sim.A * sim.om,
                height_spread=float(np.mean(rough)), height_spread_max=float(np.max(rough)),
                occupancy_spread=float(np.mean(occupancy)),
                top_wavelength=L / top, top_amplitude=float(amplitude[top]),
                peak_force=float(forces.max()),
                mean_cycle_peak_force=float(forces.max(axis=1).mean()),
                contact_fraction=float((forces > 0.05).mean()),
                impulse_alternation=float(abs(even - odd) / (even + odd)),
                max_spring_share=float(np.max(springs)),
                mean_height=float(np.mean(heights)),
                mean_kinetic_per_sphere=float(np.mean(kinetic)),
                max_overlap=max(summary["max_pair_overlap"], summary["max_plate_overlap"]),
                max_spring_speed=summary["max_spring_speed"], snaps=snaps)


if __name__ == "__main__":
    with Pool(len(CASES)) as pool:
        rows = pool.map(run, CASES)
    path = Path(__file__).resolve().parent.parent / "results" / "flat_scan.json"
    path.write_text(json.dumps(dict(runs=rows)) + "\n")
    print("wrote", path)

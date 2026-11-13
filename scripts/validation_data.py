"""Measurements of single soft contacts in the engine against the hard
(Walton) collision operator and against the linear spring-dashpot laws.

Writes results/validation.json.

    uv run python scripts/validation_data.py
"""

import json
import math
from pathlib import Path

import numpy as np

from granular import edmd as K
from granular.sim import ALPHA, EPS, KINERTIA, KT_RATIO, V0, Simulation


def paper_restitution(speed):
    return float(K.restitution(speed, EPS, ALPHA, V0, (1.0 - EPS) * V0 ** (-ALPHA)))


def place(sim, rows):
    y = 0.5 * sim.hx
    for i, (x, z, vx, vz) in enumerate(rows):
        sim.S[i, :8] = [x, y, z, vx, 0.0, vz, 0.0, 0.5]
    sim.mass[:] = 1.0
    sim.diam[:] = 1.0
    sim.W[:] = 0.0
    sim.rebuild(sort=False)


def pair(vn, psi, mode, stiffness=1e6, steps=20.0, **kwargs):
    """Two unit spheres in free fall colliding with normal closing speed vn
    and tangential surface speed psi * vn at the moment they touch, with the
    line of centres along x at that moment.  Returns what the collision did,
    resolved along that line of centres."""
    sim = Simulation(2, 20.0, 0.0, 0.3, 5.0, 1, dim=2, contact_mode=mode,
                     stiffness=stiffness, steps_per_contact=steps, **kwargs)
    lead = 0.01 / vn                       # time before the touch
    vt = psi * vn
    place(sim, [(9.5 - 0.5 * vn * lead, 30.0 - 0.5 * vt * lead, vn / 2, vt / 2),
                (10.5 + 0.5 * vn * lead, 30.0 + 0.5 * vt * lead, -vn / 2, -vt / 2)])
    status = sim.run_to(lead + 6.0 * sim.contact_time)
    assert status == 0 and sim.si[K.SI_NC] == 0
    assert sim.cnt[K.C_PP] + sim.cnt[K.C_SPRINGS] == 1
    _, v, w = sim.snapshot()
    surface = (v[1, 2] - v[0, 2]) + 0.5 * (w[0, 1] + w[1, 1])
    return dict(normal_out=float(v[1, 0] - v[0, 0]) / vn,
                psi_in=psi, psi_out=float(-surface) / vn,
                contact_time=float(sim.cnt[K.C_KICKF] * sim.P[K.P_DT]),
                max_overlap=float(sim.sf[K.SF_MAX_PAIR_OVERLAP]))


def spring_dashpot(speed, stiffness, meff=0.5):
    """Contact time and largest overlap of a linear spring-dashpot collision
    whose damping gives the paper's e(speed)."""
    loge = math.log(paper_restitution(speed))
    zeta = -loge / math.sqrt(math.pi ** 2 + loge ** 2)
    omega0 = math.sqrt(stiffness / meff)
    omegad = omega0 * math.sqrt(1.0 - zeta ** 2)
    t_peak = math.atan2(omegad, zeta * omega0) / omegad
    overlap = speed / omegad * math.exp(-zeta * omega0 * t_peak) * math.sin(omegad * t_peak)
    return math.pi / omegad, overlap


# stiff and finely stepped, so the tangential comparisons show the spring
# model itself and not the time step or the turning of the contact normal
STIFF = dict(stiffness=1e8, steps=50.0)


def main():
    out = {}
    psi = [round(x, 3) for x in np.concatenate([np.arange(0.0, 1.0, 0.125),
                                                np.arange(1.0, 8.01, 0.25)])]
    psi = [p for p in psi if p > 0.0]

    # tangential response against the hard operator, for three friction values
    out["friction"] = []
    for mu in (0.25, 0.5, 1.0):
        for p in psi:
            hard = pair(1.0, p, "never", mu=mu)
            soft = pair(1.0, p, "always", mu=mu, **STIFF)
            out["friction"].append(dict(mu=mu, psi_in=p, hard=hard["psi_out"],
                                        soft=soft["psi_out"],
                                        hard_normal=hard["normal_out"],
                                        soft_normal=soft["normal_out"]))

    # ... and for several tangential stiffnesses at the paper's mu
    out["kt_ratio"] = []
    for ratio in (0.08, KT_RATIO, 0.2, 2.0 / 7.0):
        for p in psi:
            soft = pair(1.0, p, "always", kt_ratio=ratio, **STIFF)
            out["kt_ratio"].append(dict(kt_ratio=ratio, psi_in=p, soft=soft["psi_out"]))
    out["kt_ratio_hard"] = [dict(psi_in=p, hard=pair(1.0, p, "never")["psi_out"])
                            for p in psi]

    # ... and for several impact speeds (restitution changes with speed)
    out["speed_tangential"] = []
    for vn in (0.1, 1.0, 4.0):
        for p in psi:
            out["speed_tangential"].append(dict(
                vn=vn, psi_in=p, hard=pair(vn, p, "never")["psi_out"],
                soft=pair(vn, p, "always", **STIFF)["psi_out"]))

    # normal restitution against the paper's law
    out["restitution"] = []
    for vn in np.geomspace(0.01, 10.0, 16):
        row = dict(speed=float(vn), law=paper_restitution(float(vn)))
        for steps in (20.0, 50.0):
            row["soft_%d" % steps] = pair(float(vn), 0.0, "always", steps=steps)["normal_out"]
        out["restitution"].append(row)

    # contact time and largest overlap against the spring-dashpot solution
    out["contact"] = []
    for stiffness in (1e4, 1e5, 1e6, 1e7, 1e8):
        for vn in (0.1, 1.0, 4.0):
            measured = pair(vn, 0.0, "always", stiffness=stiffness, steps=50.0)
            t_law, overlap_law = spring_dashpot(vn, stiffness)
            out["contact"].append(dict(
                stiffness=stiffness, speed=vn, contact_time=measured["contact_time"],
                contact_time_law=t_law, max_overlap=measured["max_overlap"],
                max_overlap_law=overlap_law))

    # restitution error against the time step
    out["steps"] = []
    for steps in (5, 7, 10, 14, 20, 28, 40, 56, 80, 113, 160):
        errors = []
        for vn in (0.3, 1.0, 3.0):
            errors.append(abs(pair(vn, 0.0, "always", steps=float(steps))["normal_out"]
                              / paper_restitution(vn) - 1.0))
        out["steps"].append(dict(steps=steps, max_restitution_error=max(errors)))

    # energy residual of a small layer against the time step
    out["layer_energy"] = []
    for steps in (10, 14, 20, 28, 40, 56, 80):
        sim = Simulation(100, 20.0, 5.0, 0.3, 5.0, 1, dim=2, contact_mode="tc",
                         stiffness=1e6, steps_per_contact=float(steps))
        energy0 = sim.layer_energy()[0]
        for _ in range(30):
            assert sim.run_to(sim.T) == 0
            sim.end_cycle()
        en = sim.en
        terms = [en[K.EN_PLATE_WORK], en[K.EN_PLATE_DISS], en[K.EN_PP_DISS],
                 en[K.EN_WALL_DISS], en[K.EN_SOFT_DISS]]
        out["layer_energy"].append(dict(
            steps=steps, residual=float((sim.layer_energy()[0] - energy0 - sum(terms))
                                        / sum(abs(x) for x in terms))))

    # a sphere set sliding on a static plate: slide, then roll at 5/7
    sim = Simulation(1, 40.0, 0.0, 0.3, 5.0, 1, dim=2, contact_mode="always",
                     stiffness=1e6)
    place(sim, [(5.0, 0.5 + 1e-9, 1.0, 0.0)])
    out["rolling"] = []
    for t in np.linspace(0.0, 1.0, 51)[1:]:
        assert sim.run_to(float(t)) == 0
        _, v, w = sim.snapshot()
        slide_end = 1.0 / (0.5 * (1.0 + 1.0 / KINERTIA))
        tt = min(float(t), slide_end)
        out["rolling"].append(dict(
            t=float(t), v=float(v[0, 0]), surface=float(0.5 * w[0, 1]),
            v_law=1.0 - 0.5 * tt, surface_law=0.5 / KINERTIA * tt))

    path = Path(__file__).resolve().parent.parent / "results" / "validation.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()

"""Soft contacts (springs applied as kicks) in the event-driven engine, in 2D."""

import numpy as np
import pytest

from granular import edmd as K
from granular.sim import ALPHA, BETA0, EPS, MU, V0, Simulation


def paper_restitution(speed):
    return K.restitution(speed, EPS, ALPHA, V0, (1.0 - EPS) * V0 ** (-ALPHA))


def place(sim, rows, spin=0.0):
    """Overwrite the state with unit spheres: rows of (x, z, vx, vz)."""
    y = 0.5 * sim.hx
    for i, (x, z, vx, vz) in enumerate(rows):
        sim.S[i, :8] = [x, y, z, vx, 0.0, vz, 0.0, 0.5]
    sim.mass[:] = 1.0
    sim.diam[:] = 1.0
    sim.W[:] = 0.0
    sim.W[:, 1] = spin
    sim.rebuild(sort=False)


def collide_pair(vn, vt, mode, stiffness=1e8, **kwargs):
    """Two unit spheres in free fall meeting along x with closing speed vn
    and tangential (z) relative speed vt.  Returns velocities and spins after
    the collision."""
    sim = Simulation(2, 20.0, 0.0, 0.3, 5.0, 1, dim=2, contact_mode=mode,
                     stiffness=stiffness, steps_per_contact=50, **kwargs)
    gap = 0.01
    place(sim, [(9.5 - gap / 2, 30.0, vn / 2, vt / 2),
                (10.5 + gap / 2, 30.0, -vn / 2, -vt / 2)])
    assert sim.run_to(2.0 * gap / vn + 5.0 * sim.contact_time) == 0
    if mode != "never":
        assert sim.cnt[K.C_SPRINGS] == 1 and sim.si[K.SI_NC] == 0
        assert sim.cnt[K.C_PP] == 0
    _, v, w = sim.snapshot()
    return v, w


def books(sim):
    en = sim.en
    terms = [en[K.EN_PLATE_WORK], en[K.EN_PLATE_DISS], en[K.EN_PP_DISS],
             en[K.EN_WALL_DISS], en[K.EN_SOFT_DISS]]
    return sum(terms), sum(abs(x) for x in terms)


@pytest.mark.parametrize("speed", [0.1, 1.0, 4.0])
def test_soft_pair_collision_has_paper_restitution(speed):
    v, w = collide_pair(speed, 0.0, "always")
    assert v[1, 0] - v[0, 0] == pytest.approx(paper_restitution(speed) * speed,
                                              rel=5e-3)
    assert np.all(w == 0.0)


@pytest.mark.parametrize("vn", [0.3, 1.0])
def test_soft_sliding_collision_matches_walton(vn):
    """A grazing collision slides throughout: both models apply mu times the
    normal impulse."""
    vh, wh = collide_pair(vn, 5.0 * vn, "never")
    vs, ws = collide_pair(vn, 5.0 * vn, "always")
    assert np.allclose(vs, vh, rtol=0.0, atol=5e-3 * vn)
    assert np.allclose(ws, wh, rtol=0.0, atol=1e-2 * vn)
    assert abs(wh[0, 1]) > MU * vn       # the collision did spin the spheres


@pytest.mark.parametrize("mu", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("ratio", [0.1, 0.5, 1.0, 2.0, 3.0, 4.5, 6.0])
def test_soft_collision_matches_walton_at_every_angle(mu, ratio):
    """With the Walton collision phase an isolated spring collision ends with
    the hard operator's velocities and spins, in the rolling regime, the
    sliding regime and between."""
    vn = 1.0
    vh, wh = collide_pair(vn, ratio * vn, "never", mu=mu)
    vs, ws = collide_pair(vn, ratio * vn, "always", mu=mu)
    # the line of centres turns by about ratio * vn * contact time during a
    # spring contact, which mixes the components at that order
    tolerance = 3e-3 * vn * (1.0 + ratio)
    assert np.allclose(vs, vh, rtol=0.0, atol=tolerance)
    assert np.allclose(ws, wh, rtol=0.0, atol=2.0 * tolerance)


@pytest.mark.parametrize("vt", [0.02, 0.1])
def test_tangential_spring_alone_reverses_surface_velocity_by_beta0(vt):
    """With the spring-only law, at near-normal incidence the tangential
    spring returns the surface velocity reversed by about the paper's beta0
    (KT_RATIO is set for this)."""
    v, w = collide_pair(1.0, vt, "always", tangential="spring")
    surface_before = -vt
    # n = +x, so n x (r_a w_a + r_b w_b) has z component r (w_ay + w_by)
    surface_after = (v[1, 2] - v[0, 2]) + 0.5 * (w[0, 1] + w[1, 1])
    assert -surface_after / surface_before == pytest.approx(BETA0, abs=0.04)


def test_sphere_sliding_on_the_plate_ends_rolling():
    """Coulomb friction on a spring contact: a sphere set sliding at v0 on a
    static plate ends rolling at v0 / (1 + K) = 5/7 v0, and the books show
    the energy it lost as friction."""
    sim = Simulation(1, 40.0, 0.0, 0.3, 5.0, 1, dim=2, contact_mode="always",
                     stiffness=1e6)
    place(sim, [(5.0, 0.5 + 1e-9, 1.0, 0.0)])
    energy0, _ = sim.layer_energy()
    assert sim.run_to(2.0) == 0
    _, v, w = sim.snapshot()
    assert sim.si[K.SI_NC] == 1
    assert v[0, 0] == pytest.approx(5.0 / 7.0, rel=1e-3)
    assert 0.5 * w[0, 1] == pytest.approx(5.0 / 7.0, rel=1e-3)
    energy1, _ = sim.layer_energy()
    accounted, _ = books(sim)
    assert energy1 - energy0 == pytest.approx(-1.0 / 7.0, rel=1e-3)
    assert accounted == pytest.approx(energy1 - energy0, abs=1e-5)


@pytest.mark.parametrize("speed", [0.3, 2.0])
def test_soft_plate_bounce_has_paper_restitution(speed):
    sim = Simulation(1, 20.0, 0.0, 0.3, 5.0, 1, dim=2, contact_mode="always",
                     stiffness=1e8, steps_per_contact=50)
    place(sim, [(5.0, 0.5 + 0.5 * speed ** 2, 0.0, 0.0)])
    wait = 3.0 * sim.contact_time
    assert sim.run_to(speed + wait) == 0
    _, v, _ = sim.snapshot()
    assert sim.cnt[K.C_SPRINGS] == 1 and sim.si[K.SI_NC] == 0
    assert v[0, 2] == pytest.approx(paper_restitution(speed) * speed,
                                    abs=wait + 1.5e-2 * speed)


def layer(mode, **kwargs):
    return Simulation(100, 20.0, 5.0, 0.3, 5.0, 1, dim=2, contact_mode=mode,
                      stiffness=1e6, **kwargs)


def run_cycles(sim, cycles, max_events=20_000_000):
    for _ in range(cycles):
        assert sim.run_to(sim.T, max_events=max_events) == 0
        sim.end_cycle()


@pytest.mark.parametrize("mode", ["tc", "always"])
def test_layer_runs_with_springs_and_balanced_books(mode):
    sim = layer(mode)
    energy0, _ = sim.layer_energy()
    run_cycles(sim, 12)
    energy1, _ = sim.layer_energy()
    accounted, throughput = books(sim)
    summary = sim.soft_summary()
    assert summary["springs_made"] > 1000
    if mode == "tc":
        assert sim.cnt[K.C_PP] > 1000
    else:
        assert sim.cnt[K.C_PP] == 0 and sim.cnt[K.C_PLATE] == 0
    assert max(summary["max_pair_overlap"], summary["max_plate_overlap"]) < 0.02
    assert abs(energy1 - energy0 - accounted) / throughput < 5e-3
    # the queue and cells are consistent, and no springless pair overlaps
    r, _, _ = sim.snapshot()
    assert K.check_cells(r, sim.cellof, sim.P, sim.IP) == 0
    assert K.check_events(sim.S, sim.ev_t, sim.ev_type, sim.ev_p, sim.N,
                          sim.t_local) == 0


def test_speed_limit_keeps_fast_touches_hard():
    sim = layer("tc", soft_speed_limit=0.1)
    run_cycles(sim, 6)
    summary = sim.soft_summary()
    assert summary["springs_made"] > 100
    assert summary["max_spring_speed"] <= 0.1


def test_checkpoint_restart_with_live_springs(tmp_path):
    """A run continued after saving and a run restarted from the file are
    identical, with springs alive at the checkpoint."""
    path = str(tmp_path / "state.npz")
    first = layer("tc")
    run_cycles(first, 3)
    # stop where part of the layer is on springs
    for step in range(1, 400):
        if first.si[K.SI_NC] >= 3:
            break
        if step % 100 == 0:
            assert first.run_to(first.T) == 0
            first.end_cycle()
        else:
            assert first.run_to((step % 100) * 0.01 * first.T) == 0
    assert first.si[K.SI_NC] >= 3
    first.save_checkpoint(path)
    second = layer("tc")
    second.load_checkpoint(path)
    assert second.si[K.SI_NC] == first.si[K.SI_NC]
    for sim in (first, second):
        assert sim.run_to(sim.T) == 0
        sim.end_cycle()
        run_cycles(sim, 2)
    for a, b in zip(first.snapshot(), second.snapshot()):
        assert np.array_equal(a, b)
    assert np.array_equal(first.cxi, second.cxi)


def test_never_mode_makes_no_springs():
    sim = layer("never")
    run_cycles(sim, 3)
    assert sim.cnt[K.C_SPRINGS] == 0 and sim.cnt[K.C_KICKS] == 0
    assert sim.en[K.EN_SOFT_DISS] == 0.0

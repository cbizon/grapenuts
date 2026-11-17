"""The position nudge: after a plate event the sphere sits at least two
floating-point steps above the plate, and the energy added is accounted."""

import numpy as np

from granular import edmd as K


def _state(z, r=0.5):
    S = np.zeros((2, 8))
    S[0, K.SZ] = z
    S[0, K.SR] = r
    return S


def test_raises_sphere_at_surface_by_two_steps_and_counts_energy():
    A, om, g, t = 0.7, 1.3, 1.0, 0.4
    surface = A * np.sin(om * t) + 0.5
    S = _state(surface)
    mass = np.array([1.2])
    en = np.zeros(K.NEN)
    K.nudge_onto_plate(0, t, S, mass, A, om, g, en)
    target = np.nextafter(np.nextafter(surface, np.inf), np.inf)
    assert S[0, K.SZ] == target
    assert en[K.EN_NUDGE] == 1.2 * g * (target - surface)
    assert en[K.EN_NUDGE] > 0.0


def test_raises_sphere_slightly_below_surface():
    A, om, g, t = 0.7, 1.3, 1.0, 0.4
    surface = A * np.sin(om * t) + 0.5
    S = _state(surface - 1e-12)
    en = np.zeros(K.NEN)
    K.nudge_onto_plate(0, t, S, np.array([1.0]), A, om, g, en)
    assert S[0, K.SZ] > surface
    assert np.isclose(en[K.EN_NUDGE], 1e-12, rtol=1e-3)


def test_never_lowers_a_sphere_above_target():
    A, om, g, t = 0.7, 1.3, 1.0, 0.4
    surface = A * np.sin(om * t) + 0.5
    S = _state(surface + 1e-9)
    en = np.zeros(K.NEN)
    K.nudge_onto_plate(0, t, S, np.array([1.0]), A, om, g, en)
    assert S[0, K.SZ] == surface + 1e-9
    assert en[K.EN_NUDGE] == 0.0


def test_nudge_is_off_by_default_and_on_when_asked():
    from granular.sim import Simulation

    def plate_hits(**kwargs):
        sim = Simulation(100, 20.0, 5.0, 0.3, 5.0, 1, dim=2, **kwargs)
        for _ in range(3):
            assert sim.run_to(sim.T, max_events=20_000_000) == 0
            sim.end_cycle()
        return sim

    off = plate_hits()
    on = plate_hits(plate_nudge=True)
    assert off.en[K.EN_NUDGE] == 0.0
    assert on.cnt[K.C_PLATE] > 100 and on.en[K.EN_NUDGE] > 0.0

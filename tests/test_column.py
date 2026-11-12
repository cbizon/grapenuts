"""Tests of the one-dimensional hybrid hard/soft column prototype."""

import math

import numpy as np
import pytest

from granular import column as C
from granular import edmd as K
from granular.sim import ALPHA, EPS, V0


def paper_restitution(speed):
    return K.restitution(speed, EPS, ALPHA, V0, (1.0 - EPS) * V0 ** (-ALPHA))


def dropped_sphere(speed, stiffness, mode, **kwargs):
    """One sphere released at rest above a static plate from the height that
    gives it the impact speed ``speed``.  Returns (column, impact time)."""
    column = C.Column([1.0], 0.0, 1.0, stiffness, mode, gap=0.5 * speed ** 2,
                      **kwargs)
    return column, speed


@pytest.mark.parametrize("speed", [0.01, 0.3, 2.0])
def test_hard_bounce_uses_paper_restitution(speed):
    column, t_impact = dropped_sphere(speed, 1e6, C.NEVER)
    after = 1e-6
    assert column.run_to(t_impact + after)
    assert column.stats[C.S_HARD_PLATE] == 1
    assert column.v[0] + after == pytest.approx(paper_restitution(speed) * speed,
                                                rel=1e-9)


@pytest.mark.parametrize("speed", [0.3, 1.0, 3.0])
def test_soft_bounce_reproduces_paper_restitution(speed):
    """An isolated spring collision rebounds with e(v) v and its dashpot
    dissipates what the hard collision would."""
    column, t_impact = dropped_sphere(speed, 1e8, C.ALWAYS, steps_per_contact=50)
    wait = 3.0 * column.contact_time
    assert column.run_to(t_impact + wait)
    assert not column.active[0]
    assert column.stats[C.S_SPRINGS_MADE] == 1
    e = paper_restitution(speed)
    # gravity acts for at most `wait` after the rebound
    assert column.v[0] == pytest.approx(e * speed, abs=wait + 0.01 * speed)
    assert column.acc[C.A_DASH_DISS] == pytest.approx(
        -0.5 * speed ** 2 * (1.0 - e * e), rel=0.02)
    assert column.stats[C.S_MAX_PLATE_OVERLAP] == pytest.approx(
        speed * math.sqrt(1.0 / 1e8), rel=0.2)


def test_resting_stack_carries_its_weight_on_springs():
    """On a static plate a soft stack settles with each contact compressed by
    the weight above it divided by the stiffness."""
    stiffness = 1e5
    diam = np.array([1.0, 0.99, 1.01, 1.0, 0.995])
    column = C.Column(diam, 0.0, 1.0, stiffness, C.ALWAYS, gap=1e-6)
    assert column.run_to(40.0)
    samples = []
    for step in range(200):
        column.run_to(40.0 + 0.37 * column.contact_time * (step + 1))
        samples.append(column.overlaps())
    weight_above = np.cumsum((diam ** 3)[::-1])[::-1]
    assert column.active.all()
    assert np.mean(samples, axis=0) == pytest.approx(weight_above / stiffness,
                                                     rel=0.01)


def driven(n, mode, stiffness=6e5, **kwargs):
    return C.driven_column(n, 5.0, 0.44, stiffness, mode, **kwargs)


def residual(column, energy0):
    throughput = np.abs(column.acc[:3]).sum()
    return (column.energy() - energy0 - column.accounted()) / throughput


def test_hard_column_stalls_and_hybrid_survives():
    hard, period = driven(10, C.NEVER)
    assert not hard.run_to(5 * period, max_touches=1_000_000)

    hybrid, period = driven(10, C.TC)
    energy0 = hybrid.energy()
    assert hybrid.run_to(20 * period, max_touches=1_000_000)
    assert hybrid.stats[C.S_SPRINGS_MADE] > 0
    assert hybrid.stats[C.S_HARD_PAIR] + hybrid.stats[C.S_HARD_PLATE] > 0
    assert max(hybrid.stats[C.S_MAX_PAIR_OVERLAP],
               hybrid.stats[C.S_MAX_PLATE_OVERLAP]) < 0.03
    assert abs(residual(hybrid, energy0)) < 5e-3


def test_hard_energy_books_balance_to_roundoff():
    column, period = driven(2, C.NEVER)
    energy0 = column.energy()
    assert column.run_to(20 * period, max_touches=1_000_000)
    assert abs(residual(column, energy0)) < 1e-12


def test_hybrid_is_the_hard_model_when_no_touch_is_unclean():
    """A single bouncing sphere never collides twice within t_c here, so the
    hybrid makes no spring and its trajectory is the hard one, bit for bit."""
    hard, period = driven(1, C.NEVER)
    hybrid, _ = driven(1, C.TC)
    for column in (hard, hybrid):
        assert column.run_to(20 * period, max_touches=1_000_000)
    assert hybrid.stats[C.S_SPRINGS_MADE] == 0
    assert hybrid.z[0] == hard.z[0] and hybrid.v[0] == hard.v[0]


def test_spring_energy_error_is_second_order_in_the_step():
    errors = []
    for steps in (20, 80):
        column, period = driven(10, C.ALWAYS, steps_per_contact=steps)
        energy0 = column.energy()
        assert column.run_to(50 * period)
        errors.append(abs(residual(column, energy0)))
    assert errors[1] < errors[0] / 8.0


def test_speed_limit_keeps_fast_touches_hard():
    limit = 0.1
    column, period = driven(10, C.TC, soft_speed_limit=limit)
    assert column.run_to(20 * period, max_touches=5_000_000)
    assert column.stats[C.S_SPRINGS_MADE] > 0
    assert column.stats[C.S_MAX_SPRING_SPEED] <= limit

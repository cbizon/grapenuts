"""One-dimensional prototype of the hybrid hard/soft contact algorithm: a
vertical column of spheres on the oscillating plate.

Spheres are ordered bottom to top.  Contact 0 is plate-sphere 0 and contact
i is sphere (i-1)-sphere i.  Units are those of the engine (diameter D,
gravity g, mass of a sphere of diameter D), and the restitution law and the
plate-root solver are the engine's.

Algorithm:

* between events every sphere is in free flight under gravity;
* a **touch** event is the instant a contact without a spring closes.  If the
  touch is *clean* it is an ordinary hard collision with the paper's e(v).
  Otherwise the contact gets a spring, starting at zero compression, and
  velocities are left alone;
* every ``dt`` a **kick** gives each spring's two spheres the impulse
  ``(k delta + gamma ddelta/dt) dt``.  A spring found open (delta <= 0) at a
  kick is removed.

Which touches are clean is the trigger:

* ``NEVER``: all of them (the hard-sphere model);
* ``ALWAYS``: none (soft spheres);
* ``TC``: a touch is clean unless either partner collided, or lost a spring,
  within the last ``tc_window``, or holds a spring now.  With a finite
  ``soft_speed_limit`` a touch faster than that is clean regardless, so
  springs only form at low approach speed.

A spring's damping is fixed when it is created, from the approach speed v at
the touch, so that an isolated linear spring-dashpot collision has the
paper's restitution e(v): zeta = -ln e / sqrt(pi^2 + ln^2 e).  The force is
not clamped at zero, which is what makes that relation exact.
"""

import math
from dataclasses import dataclass, field

import numpy as np
from numba import njit

from . import edmd as K
from .sim import ALPHA, EPS, V0, drive_parameters

NEVER, ALWAYS, TC = 0, 1, 2

# float parameters
Q_A, Q_OM, Q_G, Q_EPS, Q_ALPHA, Q_V0, Q_B, Q_K, Q_DT, Q_TCW, Q_VSOFT = range(11)
# accounting: plate work, dissipation in hard collisions, dissipation in
# dashpots, and the time integral of the centre-of-mass height
A_PLATE_WORK, A_HARD_DISS, A_DASH_DISS, A_ZCM_INT = range(4)
# statistics
(S_HARD_PAIR, S_HARD_PLATE, S_SPRINGS_MADE, S_KICK_FORCES, S_KICK_TIMES,
 S_MAX_PAIR_OVERLAP, S_MAX_PLATE_OVERLAP, S_MAX_SPRING_SPEED) = range(8)

HARD_OVERLAP_TOL = 2e-8     # a springless pair deeper than this is a bug


@njit(cache=True)
def _run(z, v, r, mass, active, gamma, last, Q, mode, t, kick_index, t_end,
         max_touches, acc, stats):
    """Advance to t_end.  Returns (status, t, kick_index); status 1 means
    max_touches touch events were used up first (a stall)."""
    n = z.shape[0]
    A = Q[Q_A]
    om = Q[Q_OM]
    g = Q[Q_G]
    k = Q[Q_K]
    dt = Q[Q_DT]
    tcw = Q[Q_TCW]
    total_mass = mass.sum()
    dvel = np.zeros(n)
    dash = np.zeros(n)      # dashpot impulse of each contact in this kick
    touches = 0
    while True:
        kick_time = kick_index * dt
        is_kick = kick_time <= t_end
        target = kick_time if is_kick else t_end
        horizon = target - t
        best = horizon
        which = -1
        if horizon > 0.0:
            if not active[0]:
                tau = K.plate_root(z[0], v[0], r[0], t, horizon, A, om, g)
                if tau < best:
                    best = tau
                    which = 0
            for i in range(1, n):
                if not active[i]:
                    closing = v[i - 1] - v[i]
                    if closing > 0.0:
                        gap = z[i] - z[i - 1] - r[i] - r[i - 1]
                        if gap < -HARD_OVERLAP_TOL:
                            raise ValueError("springless pair overlaps")
                        tau = max(gap, 0.0) / closing
                        if tau < best:
                            best = tau
                            which = i
            for i in range(n):
                z[i] += (v[i] - 0.5 * g * best) * best
                v[i] -= g * best
        if which < 0:
            t = target
        else:
            t += best

        if which >= 0:
            touches += 1
            if touches > max_touches:
                return 1, t, kick_index
            c = which
            vp = A * om * math.cos(om * t)
            if c == 0:
                closing = vp - v[0]
                meff = mass[0]
            else:
                closing = v[c - 1] - v[c]
                meff = mass[c - 1] * mass[c] / (mass[c - 1] + mass[c])
            if closing <= 0.0:
                continue
            soft = mode == ALWAYS
            if mode == TC:
                soft = (t - last[c] <= tcw or active[c]
                        or (c + 1 < n and active[c + 1]))
                if c > 0 and not soft:
                    soft = (t - last[c - 1] <= tcw or active[c - 1])
                soft = soft and closing <= Q[Q_VSOFT]
            e = K.restitution(closing, Q[Q_EPS], Q[Q_ALPHA], Q[Q_V0], Q[Q_B])
            if soft:
                active[c] = True
                loge = math.log(e)
                zeta = -loge / math.sqrt(math.pi ** 2 + loge ** 2)
                gamma[c] = 2.0 * zeta * math.sqrt(k * meff)
                stats[S_SPRINGS_MADE] += 1
                if closing > stats[S_MAX_SPRING_SPEED]:
                    stats[S_MAX_SPRING_SPEED] = closing
            else:
                impulse = meff * (1.0 + e) * closing
                acc[A_HARD_DISS] += 0.5 * meff * closing * closing * (e * e - 1.0)
                last[c] = t
                if c == 0:
                    v[0] += impulse / mass[0]
                    acc[A_PLATE_WORK] += vp * impulse
                    stats[S_HARD_PLATE] += 1
                else:
                    v[c - 1] -= impulse / mass[c - 1]
                    v[c] += impulse / mass[c]
                    last[c - 1] = t
                    stats[S_HARD_PAIR] += 1
            continue

        if not is_kick:
            return 0, t, kick_index

        # kick: all forces from the same positions and velocities
        zp = A * math.sin(om * t)
        vp = A * om * math.cos(om * t)
        for i in range(n):
            dvel[i] = 0.0
        for c in range(n):
            if not active[c]:
                continue
            if c == 0:
                delta = zp + r[0] - z[0]
                rate = vp - v[0]
            else:
                delta = r[c] + r[c - 1] - (z[c] - z[c - 1])
                rate = v[c - 1] - v[c]
            dash[c] = 0.0
            if delta <= 0.0:
                active[c] = False
                last[c] = t
                if c > 0:
                    last[c - 1] = t
                continue
            dash[c] = gamma[c] * rate * dt
            impulse = k * delta * dt + dash[c]
            stats[S_KICK_FORCES] += 1
            if c == 0:
                dvel[0] += impulse / mass[0]
                acc[A_PLATE_WORK] += vp * impulse
                if delta > stats[S_MAX_PLATE_OVERLAP]:
                    stats[S_MAX_PLATE_OVERLAP] = delta
            else:
                dvel[c - 1] -= impulse / mass[c - 1]
                dvel[c] += impulse / mass[c]
                if delta > stats[S_MAX_PAIR_OVERLAP]:
                    stats[S_MAX_PAIR_OVERLAP] = delta
        zcm = 0.0
        for i in range(n):
            v[i] += dvel[i]
            zcm += mass[i] * z[i]
        # an impulse J on a contact changes the kinetic energy in the
        # contact's frame by -J times the mean of the closing rate before
        # and after the kick; the dashpot's share of that is dissipation
        for c in range(n):
            if active[c]:
                if c == 0:
                    rate = vp - v[0]
                    before = rate + dvel[0]
                else:
                    rate = v[c - 1] - v[c]
                    before = rate - dvel[c - 1] + dvel[c]
                acc[A_DASH_DISS] -= dash[c] * 0.5 * (before + rate)
        acc[A_ZCM_INT] += zcm / total_mass * dt
        stats[S_KICK_TIMES] += 1
        kick_index += 1


@dataclass
class Column:
    """State of a column of spheres with diameters ``diam`` (bottom to top),
    starting at rest with gaps ``gap`` between neighbours and above the plate
    at its t = 0 position.

    ``stiffness`` is k in units m g / D.  ``steps_per_contact`` is t_c / dt
    with t_c = pi sqrt(m_eff / k) for two spheres of diameter 1.  ``tc_window``
    defaults to t_c.  ``soft_speed_limit`` applies in TC mode only."""

    diam: np.ndarray
    amplitude: float
    omega: float
    stiffness: float
    mode: int
    steps_per_contact: float = 20.0
    tc_window: float | None = None
    soft_speed_limit: float = math.inf
    gap: float = 1e-3
    eps: float = EPS
    alpha: float = ALPHA
    v0: float = V0
    t: float = 0.0
    kick_index: int = 1
    z: np.ndarray = field(init=False)
    v: np.ndarray = field(init=False)

    def __post_init__(self):
        self.diam = np.asarray(self.diam, dtype=float)
        n = self.diam.size
        self.r = 0.5 * self.diam
        self.mass = self.diam ** 3
        self.z = np.cumsum(self.diam) - self.r + self.gap * np.arange(1, n + 1)
        self.v = np.zeros(n)
        self.active = np.zeros(n, dtype=np.bool_)
        self.gamma = np.zeros(n)
        self.last = np.full(n, -np.inf)
        self.contact_time = math.pi * math.sqrt(0.5 / self.stiffness)
        self.dt = self.contact_time / self.steps_per_contact
        if self.tc_window is None:
            self.tc_window = self.contact_time
        self.Q = np.array([self.amplitude, self.omega, 1.0, self.eps,
                           self.alpha, self.v0,
                           (1.0 - self.eps) * self.v0 ** (-self.alpha),
                           self.stiffness, self.dt, self.tc_window,
                           self.soft_speed_limit])
        self.acc = np.zeros(4)
        self.stats = np.zeros(8)

    def run_to(self, t_end, max_touches=2 ** 62):
        """Advance to t_end.  Returns True if it got there, False if
        max_touches touch events were used first."""
        status, self.t, self.kick_index = _run(
            self.z, self.v, self.r, self.mass, self.active, self.gamma,
            self.last, self.Q, self.mode, self.t, self.kick_index,
            float(t_end), max_touches, self.acc, self.stats)
        return status == 0

    def overlaps(self):
        """Compression of every contact (negative where open)."""
        delta = np.empty(self.diam.size)
        delta[0] = self.amplitude * math.sin(self.omega * self.t) + self.r[0] - self.z[0]
        delta[1:] = self.r[1:] + self.r[:-1] - np.diff(self.z)
        return delta

    def energy(self):
        """Kinetic plus gravitational plus spring energy."""
        delta = np.where(self.active, self.overlaps(), 0.0)
        return float(0.5 * np.sum(self.mass * self.v ** 2)
                     + np.sum(self.mass * self.z)
                     + 0.5 * self.stiffness * np.sum(np.maximum(delta, 0.0) ** 2))

    def accounted(self):
        """Energy the books say went in: plate work plus (negative)
        dissipation in hard collisions and dashpots."""
        return float(self.acc[A_PLATE_WORK] + self.acc[A_HARD_DISS]
                     + self.acc[A_DASH_DISS])


def driven_column(n, gamma, f_star, stiffness, mode, seed=1, **kwargs):
    """Column of n spheres with diameters uniform in [0.99, 1.01], driven as
    a layer n deep at (gamma, f_star).  Returns (column, period)."""
    _, om, A, period = drive_parameters(gamma, f_star, float(n))
    diam = 0.99 + 0.02 * np.random.default_rng(seed).random(n)
    return Column(diam, A, om, stiffness, mode, **kwargs), period

"""Driver for the event-driven simulation of Bizon et al. (1998)."""

import math
import os

import numpy as np
from numba import njit

from . import edmd as K


# model constants from the paper
EPS = 0.7          # restitution coefficient at high impact speed
ALPHA = 0.75       # exponent of e(vn) = 1 - B vn^alpha
V0 = 1.0           # crossover velocity sqrt(g D)
MU = 0.5           # sliding friction coefficient
BETA0 = 0.35       # rolling-limit rotational restitution
KINERTIA = 0.4     # 4 I / (m sigma^2) for uniform spheres
PHI = 0.58         # packing fraction used to define the layer depth

# soft contacts (not in the paper)
KT_RATIO = 0.125   # tangential / normal spring stiffness; gives beta0 = 0.35 (measured)
CONTACT_MODES = {"never": K.MODE_NEVER, "always": K.MODE_ALWAYS, "tc": K.MODE_TC}


def drive_parameters(gamma, f_star, depth):
    """Frequency, angular frequency, amplitude and period in units D, g."""
    f = f_star / math.sqrt(depth)          # f* = f sqrt(H/g), H = N D
    om = 2.0 * math.pi * f
    A = gamma / om ** 2                    # Gamma = 4 pi^2 f^2 A / g
    return f, om, A, 1.0 / f


@njit(cache=True)
def _rsa_fill(n, L, zlo, zhi, rad, seed, out, hc, ncx, ncz, maxtry, y_plane):
    """Random sequential addition of n spheres (radii rad) in the slab
    [r, L-r]^2 x [zlo + r, zhi - r] without overlaps (gap >= 1e-3).

    If y_plane >= 0, every particle is placed at y = y_plane (spheres in the
    x-z plane) and y is not drawn."""
    np.random.seed(seed)
    ncy = 1 if y_plane >= 0.0 else ncx
    head = -np.ones(ncx * ncy * ncz, dtype=np.int64)
    nxt = -np.ones(n, dtype=np.int64)
    placed = 0
    tries = 0
    while placed < n:
        tries += 1
        if tries > maxtry:
            return placed
        ri = rad[placed]
        x = ri + (L - 2 * ri) * np.random.random()
        if y_plane >= 0.0:
            y = y_plane
        else:
            y = ri + (L - 2 * ri) * np.random.random()
        z = zlo + ri + (zhi - zlo - 2 * ri) * np.random.random()
        ix = min(int(x / hc), ncx - 1)
        iy = min(int(y / hc), ncy - 1)
        iz = min(int((z - zlo) / hc), ncz - 1)
        ok = True
        for jz in range(max(iz - 1, 0), min(iz + 2, ncz)):
            for jy in range(max(iy - 1, 0), min(iy + 2, ncy)):
                for jx in range(max(ix - 1, 0), min(ix + 2, ncx)):
                    j = head[jx + ncx * (jy + ncy * jz)]
                    while j >= 0:
                        dx = out[j, 0] - x
                        dy = out[j, 1] - y
                        dz = out[j, 2] - z
                        s = ri + rad[j] + 1e-3
                        if dx * dx + dy * dy + dz * dz < s * s:
                            ok = False
                            break
                        j = nxt[j]
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            out[placed, 0] = x
            out[placed, 1] = y
            out[placed, 2] = z
            c = ix + ncx * (iy + ncy * iz)
            nxt[placed] = head[c]
            head[c] = placed
            placed += 1
    return placed


class Simulation:
    """State and driver of one simulation run (lengths in D, time in sqrt(D/g)).

    ``dim=3`` simulates spheres in an L x L box.  ``dim=2`` simulates spheres
    confined to a plane, as in the 1998 C code's 2D mode: every sphere sits on
    the plane y = hx / 2 with zero y velocity and spin only about y, in an x-z
    box of width L.  The 3D kernel then keeps the motion exactly planar (all
    impulses lie in the plane, all torques along y).  Masses and moments of
    inertia are those of spheres.  The y side walls are never reached.

    Internally particles are re-ordered for memory locality; ``pid[k]`` is the
    original (exported) index of internal particle k.

    Soft contacts: ``contact_mode`` chooses which touches become springs
    instead of hard collisions, ``"never"`` (the paper's hard-sphere model),
    ``"always"``, or ``"tc"`` (see ``edmd.touch_is_soft``).  ``stiffness`` is
    the normal spring constant k in units m g / D; ``kt_ratio`` the
    tangential one relative to it; the springs act as impulses every
    t_c / ``steps_per_contact``, with t_c = pi sqrt(m_eff / k) for two
    spheres of diameter 1.  ``tc_window`` (default t_c) and
    ``soft_speed_limit`` belong to the ``"tc"`` trigger.  Side walls are
    always hard.
    """

    def __init__(self, n_particles, box_width, gamma, f_star, depth, seed,
                 box_height=None, fill_fraction=0.30, v_init=0.1,
                 btol=1e-10, mu=MU,
                 eps=EPS, alpha=ALPHA, v0=V0, dim=3,
                 contact_mode="never", stiffness=None, steps_per_contact=20.0,
                 kt_ratio=KT_RATIO, tc_window=None, soft_speed_limit=math.inf):
        if dim not in (2, 3):
            raise ValueError("dim must be 2 or 3, got %r" % (dim,))
        if contact_mode not in CONTACT_MODES:
            raise ValueError("unknown contact_mode %r" % (contact_mode,))
        if contact_mode != "never" and stiffness is None:
            raise ValueError("contact_mode %r needs a stiffness" % (contact_mode,))
        self.contact_mode = contact_mode
        self.dim = int(dim)
        self.N = int(n_particles)
        self.L = float(box_width)
        self.gamma = float(gamma)
        self.f_star = float(f_star)
        self.depth = float(depth)
        self.seed = int(seed)
        self.f, self.om, self.A, self.T = drive_parameters(gamma, f_star, depth)
        rng = np.random.default_rng(self.seed)

        # diameters uniform in [0.99, 1.01] (stratified so that mean = 1)
        u = (rng.permutation(self.N) + rng.random(self.N)) / self.N
        diam = 0.99 + 0.02 * u
        self.rmax = float(0.5 * diam.max())

        # cell grid: square cells of side >= max diameter
        nx = int(math.floor(self.L / 1.0105))
        self.nx = nx
        self.ny = nx if dim == 3 else 1
        self.hx = self.L / nx
        self.hz = self.hx
        self.z0 = -self.A - self.hz + 0.25     # layer 0 always empty
        if box_height is None:
            box_height = 60.0
        self.box_height = float(box_height)
        self.nz = int(math.ceil((self.box_height - self.z0) / self.hz)) + 1

        # initial positions: random sequential addition in a loose slab
        if dim == 3:
            vol = np.sum(math.pi / 6.0 * diam ** 3)
            cross_section = self.L ** 2
            y_plane = -1.0
        else:
            # area the spheres' cross-sections cover in the plane
            vol = np.sum(math.pi / 4.0 * diam ** 2)
            cross_section = self.L
            y_plane = 0.5 * self.hx
        r = np.zeros((self.N, 3))
        seed32 = int(rng.integers(0, 2 ** 31 - 1))
        for attempt in range(10):
            # the slab must fit at least one diameter, or RSA would place
            # spheres below zlo + r, inside the plate
            hslab = max(vol / (fill_fraction * cross_section), 2.0 * self.rmax)
            zlo = 0.0 + 1e-3
            zhi = zlo + hslab
            hc = 1.02
            ncx = int(self.L / hc) + 1
            ncz = int((zhi - zlo) / hc) + 2
            placed = _rsa_fill(self.N, self.L, zlo, zhi, 0.5 * diam, seed32,
                               r, hc, ncx, ncz, 200 * self.N, y_plane)
            if placed == self.N:
                break
            fill_fraction *= 0.9
        if placed != self.N:
            raise RuntimeError("RSA placement failed: %d of %d" % (placed, self.N))
        self.fill_fraction = fill_fraction
        v = rng.normal(0.0, v_init, size=(self.N, 3))
        if dim == 2:
            v[:, 1] = 0.0

        # array-of-structures state (x, y, z, vx, vy, vz, t_state, radius)
        self.S = np.zeros((self.N + 1, 8))
        self.S[:self.N, 0:3] = r
        self.S[:self.N, 3:6] = v
        self.S[:self.N, 7] = 0.5 * diam
        self.W = np.zeros((self.N, 3))
        self.diam = diam                  # internal order
        self.mass = diam ** 3             # uniform density, m(D) = 1
        self.pid = np.arange(self.N)

        self.P = np.zeros(K.NP_FLOAT)
        self.P[K.P_L] = self.L
        self.P[K.P_HX] = self.hx
        self.P[K.P_HZ] = self.hz
        self.P[K.P_Z0] = self.z0
        self.P[K.P_A] = self.A
        self.P[K.P_OM] = self.om
        self.P[K.P_G] = 1.0
        self.P[K.P_EPS] = eps
        self.P[K.P_ALPHA] = alpha
        self.P[K.P_V0] = v0
        self.P[K.P_B] = (1.0 - eps) * v0 ** (-alpha)
        self.P[K.P_MU] = mu
        self.P[K.P_BETA0] = BETA0
        self.P[K.P_K] = KINERTIA
        self.P[K.P_BTOL] = btol
        self.P[K.P_RMAX] = self.rmax
        if stiffness is not None:
            self.contact_time = math.pi * math.sqrt(0.5 / stiffness)
            self.P[K.P_KN] = stiffness
            self.P[K.P_KT] = kt_ratio * stiffness
            self.P[K.P_DT] = self.contact_time / steps_per_contact
            self.P[K.P_TCW] = (self.contact_time if tc_window is None
                               else tc_window)
            self.P[K.P_VSOFT] = soft_speed_limit
        M = 1
        while M < self.N + 1:
            M *= 2
        self.M = M
        self.IP = np.zeros(K.NP_INT, dtype=np.int64)
        self.IP[K.I_NX] = nx
        self.IP[K.I_NY] = self.ny
        self.IP[K.I_NZ] = self.nz
        self.IP[K.I_N] = self.N
        self.IP[K.I_M] = M
        self.IP[K.I_MODE] = CONTACT_MODES[contact_mode]

        self.cellof = np.zeros(self.N, dtype=np.int64)
        self.head = -np.ones(nx * self.ny * self.nz, dtype=np.int64)
        self.nxt = -np.ones(self.N, dtype=np.int64)
        self.prv = -np.ones(self.N, dtype=np.int64)
        self.ev_t = np.zeros(self.N + 1)
        self.ev_type = np.zeros(self.N + 1, dtype=np.int64)
        self.ev_p = np.zeros(self.N + 1, dtype=np.int64)
        self.tree = np.zeros(2 * M, dtype=np.int64)
        self.cnt = np.zeros(K.NCOUNT, dtype=np.int64)
        self.en = np.zeros(K.NEN)     # energy and impulse accounting
        # soft contacts: the spring list, each sphere's partners, the time of
        # each sphere's last collision, and scratch space for kicks
        cap = 8 * self.N + 16
        self.ca = np.zeros(cap, dtype=np.int64)
        self.cb = np.zeros(cap, dtype=np.int64)
        self.cgam = np.zeros(cap)
        self.cxi = np.zeros((cap, 3))
        self.cscr = np.zeros((cap, 6))
        self.cp = np.zeros((self.N, K.MAXC), dtype=np.int64)
        self.ncp = np.zeros(self.N, dtype=np.int64)
        self.last = np.full(self.N, -np.inf)
        self.dV = np.zeros((self.N, 3))
        self.dW = np.zeros((self.N, 3))
        self.flag = np.zeros(self.N, dtype=np.int8)
        self.touched = np.zeros(self.N, dtype=np.int64)
        self.si = np.zeros(K.NSI, dtype=np.int64)
        self.sf = np.zeros(K.NSF)
        self.cycle = 0            # completed forcing cycles
        self.t_local = 0.0        # time since start of current cycle
        self.rebuild()

    # ------------------------------------------------------------------
    @property
    def rad(self):
        return self.S[:self.N, 7]

    def _sort_particles(self):
        """Re-order particles column by column for memory locality (states
        must be synchronized)."""
        S = self.S[:self.N]
        ix = np.clip((S[:, 0] / self.hx).astype(np.int64), 0, self.nx - 1)
        iy = np.clip((S[:, 1] / self.hx).astype(np.int64), 0, self.ny - 1)
        key = (iy * self.nx + ix) * 1000.0 + (S[:, 2] - self.z0)
        perm = np.argsort(key, kind="stable")
        self.S[:self.N] = S[perm]
        self.W[:] = self.W[perm]
        self.diam = self.diam[perm]
        self.mass = self.mass[perm]
        self.pid = self.pid[perm]
        self.last = self.last[perm]
        new_index = np.empty(self.N, dtype=np.int64)
        new_index[perm] = np.arange(self.N)
        nc = self.si[K.SI_NC]
        self.ca[:nc] = new_index[self.ca[:nc]]
        sphere = self.cb[:nc] >= 0
        self.cb[:nc][sphere] = new_index[self.cb[:nc][sphere]]
        K.rebuild_partners(self.ca, self.cb, self.cp, self.ncp, self.si)

    def rebuild(self, sort=True):
        """Synchronize all particles at t_local, re-sort, and rebuild the cell
        lists and the event queue (every particle re-predicts at t_local)."""
        kick_time = self.ev_t[self.N] if self.si[K.SI_NC] > 0 else K.INF
        K.synchronize(self.t_local, self.S, 1.0, self.N)
        if sort:
            self._sort_particles()
        K.build_cells(self.S, self.cellof, self.head, self.nxt, self.prv,
                      self.P, self.IP)
        K.init_events(self.ev_t, self.ev_type, self.ev_p, self.tree, self.IP,
                      self.t_local)
        if kick_time < K.INF:
            K.restore_kick(kick_time, self.ev_t, self.ev_type, self.tree, self.IP)

    def plate(self, t_local):
        return (self.A * math.sin(self.om * t_local),
                self.A * self.om * math.cos(self.om * t_local))

    def run_to(self, t_local_stop, max_events=2 ** 62):
        st = K.run_events(self.S, self.W, self.mass, self.cellof, self.head,
                          self.nxt, self.prv, self.ev_t, self.ev_type,
                          self.ev_p, self.tree, self.P, self.IP,
                          float(t_local_stop), max_events, self.cnt, self.en,
                          self.ca, self.cb, self.cgam, self.cxi, self.cscr,
                          self.cp, self.ncp, self.last, self.dV, self.dW,
                          self.flag, self.touched, self.si, self.sf)
        if st == 0:
            self.t_local = float(t_local_stop)
        return st

    def end_cycle(self):
        """Call when t_local == T: shift the time origin by one period."""
        K.rebase(self.T, self.S, self.ev_t, self.N, self.last)
        self.t_local = 0.0
        self.cycle += 1

    def snapshot(self, original_order=False):
        """Positions, velocities, angular velocities at the current time."""
        r = np.empty((self.N, 3))
        v = np.empty((self.N, 3))
        K.project(self.t_local, self.S, 1.0, r, v)
        w = self.W.copy()
        if original_order:
            ro = np.empty_like(r)
            vo = np.empty_like(v)
            wo = np.empty_like(w)
            ro[self.pid] = r
            vo[self.pid] = v
            wo[self.pid] = w
            return ro, vo, wo
        return r, v, w

    def diameters_original(self):
        d = np.empty(self.N)
        d[self.pid] = self.diam
        return d

    def time_abs(self):
        return self.cycle * self.T + self.t_local

    # ------------------------------------------------------------------
    def diagnostics(self):
        r, v, w = self.snapshot()
        zp, vp = self.plate(self.t_local)
        rad = self.rad
        gap_plate = float(np.min(r[:, 2] - rad - zp))
        ke = 0.5 * float(np.sum(self.mass[:, None] * v ** 2))
        kr = 0.5 * float(np.sum(0.1 * self.mass[:, None] * self.diam[:, None] ** 2 * w ** 2))
        return dict(zcm=float(np.mean(r[:, 2])), zmax=float(np.max(r[:, 2])),
                    ke=ke, krot=kr, gap_plate=gap_plate,
                    xmin=float(np.min(r[:, 0] - rad)),
                    xmax=float(np.max(r[:, 0] + rad)),
                    ymin=float(np.min(r[:, 1] - rad)),
                    ymax=float(np.max(r[:, 1] + rad)))

    def layer_energy(self):
        """Kinetic, rotational, gravitational potential and spring energy of
        the layer, and its vertical momentum, at the current time."""
        r, v, w = self.snapshot()
        inertia = 0.25 * KINERTIA * self.mass * self.diam ** 2
        kinetic = 0.5 * float(np.sum(self.mass[:, None] * v ** 2))
        rotational = 0.5 * float(np.sum(inertia[:, None] * w ** 2))
        potential = float(self.P[K.P_G] * np.sum(self.mass * r[:, 2]))
        momentum_z = float(np.sum(self.mass * v[:, 2]))
        springs = K.spring_energy(self.t_local, self.S, self.ca, self.cb,
                                  self.cxi, self.si, self.P)
        return kinetic + rotational + potential + springs, momentum_z

    def soft_summary(self):
        """Counts and extremes of the soft contacts so far."""
        return dict(
            springs_now=int(self.si[K.SI_NC]),
            springs_made=int(self.cnt[K.C_SPRINGS]),
            kicks=int(self.cnt[K.C_KICKS]),
            kick_forces=int(self.cnt[K.C_KICKF]),
            max_pair_overlap=float(self.sf[K.SF_MAX_PAIR_OVERLAP]),
            max_plate_overlap=float(self.sf[K.SF_MAX_PLATE_OVERLAP]),
            max_spring_speed=float(self.sf[K.SF_MAX_SPRING_SPEED]))

    def overlap_check(self):
        """Minimum pair gap with states projected to the current time
        (does not modify the dynamical state)."""
        r, v, w = self.snapshot()
        g, i, j = K.min_gap(r, self.rad.copy(), self.cellof, self.head,
                            self.nxt, self.P, self.IP)
        bad = K.check_cells(r, self.cellof, self.P, self.IP)
        bad += K.check_events(self.S, self.ev_t, self.ev_type, self.ev_p,
                              self.N, self.t_local)
        return g, i, j, bad

    # ------------------------------------------------------------------
    def save_checkpoint(self, path):
        """Synchronize all particles at the current time, save, and rebuild the
        event queue (so a restarted and a continued run are identical)."""
        self.rebuild(sort=True)
        nc = self.si[K.SI_NC]
        tmp = path + ".tmp.npz"
        np.savez(tmp, S=self.S, W=self.W, diam=self.diam, mass=self.mass,
                 pid=self.pid, cycle=self.cycle, t_local=self.t_local,
                 cnt=self.cnt, en=self.en,
                 ca=self.ca[:nc], cb=self.cb[:nc], cgam=self.cgam[:nc],
                 cxi=self.cxi[:nc], last=self.last, sf=self.sf,
                 kick_time=self.ev_t[self.N],
                 params=np.array([self.N, self.L, self.gamma, self.f_star,
                                  self.depth, self.seed, self.dim]))
        os.replace(tmp, path)

    def load_checkpoint(self, path):
        d = np.load(path)
        prm = d["params"]
        assert int(prm[0]) == self.N and prm[1] == self.L
        assert prm[2] == self.gamma and prm[3] == self.f_star
        dim = int(prm[6]) if len(prm) > 6 else 3
        if dim != self.dim:
            raise ValueError("checkpoint is %dD, simulation is %dD" % (dim, self.dim))
        self.S[:] = d["S"]
        self.W[:] = d["W"]
        self.diam = d["diam"].copy()
        self.mass = d["mass"].copy()
        self.pid = d["pid"].copy()
        self.cycle = int(d["cycle"])
        self.t_local = float(d["t_local"])
        self.cnt[:] = d["cnt"]
        if "en" in d.files:
            self.en[:] = d["en"]
        nc = len(d["ca"])
        self.si[K.SI_NC] = nc
        self.ca[:nc] = d["ca"]
        self.cb[:nc] = d["cb"]
        self.cgam[:nc] = d["cgam"]
        self.cxi[:nc] = d["cxi"]
        self.last[:] = d["last"]
        self.sf[:] = d["sf"]
        self.ev_t[self.N] = float(d["kick_time"])
        K.rebuild_partners(self.ca, self.cb, self.cp, self.ncp, self.si)
        self.rebuild(sort=False)

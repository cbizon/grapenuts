"""Event-driven hard-sphere kernel for the vertically oscillated granular layer
of Bizon et al., PRL 80, 57 (1998).

Model (all quantities in units of the mean diameter D, gravity g, and the mass
of a sphere of diameter D):

* particles are inelastic, frictional hard spheres that interact only through
  instantaneous binary collisions and move under gravity between collisions;
* normal restitution  e(vn) = 1 - B vn^alpha  for vn < v0,  e = eps  otherwise,
  B = (1 - eps) v0^-alpha;
* tangential impulse from Walton's frictional hard-sphere collision operator
  (Walton 1993, Eqs. 25.15-25.27): sliding with Coulomb friction mu, cut off at
  the rolling limit where the tangential surface-velocity ratio is -beta0;
* stationary side walls at x = 0, L and y = 0, L, and a bottom plate at
  z_p(t) = A sin(omega t); ball-wall collisions use the same eps, mu, beta0;
* delayed-states event-driven algorithm (Lubachevsky 1991, Fig. 6.1; Marin,
  Risso & Cordero 1993): every particle keeps its own state time and only the
  particles taking part in an event are updated; one scheduled event per
  particle, a complete binary tree over the particle event times, and a cell
  grid whose boundary crossings are (virtual-wall) events.

Particle state is stored as an array of structures S[i] = (x, y, z, vx, vy,
vz, t_state, radius) so that the neighbour loop touches one cache line per
particle; angular velocities W[i] and masses are only needed in collisions.
"""

import numpy as np
from numba import njit

# event types
EV_PP = 1      # particle-particle collision, partner = particle index
EV_WALL = 2    # side-wall collision, partner = 0:x-, 1:x+, 2:y-, 3:y+
EV_PLATE = 3   # collision with the oscillating bottom plate
EV_CELL = 4    # cell-boundary crossing, partner = face 0:x- 1:x+ 2:y- 3:y+ 4:z- 5:z+
EV_ADV = 5     # advancement: bring state to event time and re-predict
EV_KICK = 6    # spring impulses on every soft contact (scheduled in slot N)

# which touches become springs (IP[I_MODE]); see touch_is_soft
MODE_NEVER, MODE_ALWAYS, MODE_TC = 0, 1, 2
# tangential law of a spring (IP[I_TANG]): the tangential spring alone, or
# Walton's collision impulse first and the spring once the contact outlasts
# an isolated collision; see kick
TANG_SPRING, TANG_WALTON = 0, 1
# per-spring collision-phase state (cwal columns): unit direction of the
# Walton impulse on sphere a, how much of it is left to deliver, its ratio to
# the normal impulse, and how much of the collision phase is left
CW_DX, CW_DY, CW_DZ, CW_LEFT, CW_RATIO, CW_TIME = range(6)
PLATE = -2     # partner code of the bottom plate in the soft-contact lists
MAXC = 16      # soft contacts one sphere can hold

# state columns
SX, SY, SZ, SVX, SVY, SVZ, ST, SR = range(8)

# float parameter vector layout
P_L, P_HX, P_HZ, P_Z0, P_A, P_OM, P_G, P_EPS, P_ALPHA, P_V0, P_B, P_MU, \
    P_BETA0, P_K, P_BTOL, P_RMAX, P_KN, P_KT, P_DT, P_TCW, P_VSOFT = range(21)
NP_FLOAT = 21
# int parameter vector layout
I_NX, I_NY, I_NZ, I_N, I_M, I_MODE, I_TANG = range(7)
NP_INT = 7

# counters
C_PP, C_WALL, C_PLATE, C_CELL, C_ADV, C_EVENTS, C_PPSKIP, C_PLSKIP, \
    C_TINYPP, C_TINYPL, C_PREDICT, C_PAIRS, C_ROOTCALL, C_EGAIN, C_SPRINGS, \
    C_KICKS, C_KICKF = range(17)
NCOUNT = 20

# energy and impulse accounting, accumulated over all events (see run_events)
EN_PLATE_WORK, EN_PLATE_DISS, EN_PP_DISS, EN_WALL_DISS, EN_PLATE_JZ, \
    EN_WALL_JZ, EN_MAX_GAIN, EN_SOFT_DISS = range(8)
NEN = 8
# soft-contact state: integers (number of springs) and floats (largest pair
# and plate overlap seen at a kick, largest approach speed at spring creation,
# and the total of any positive work done by friction impulses)
SI_NC = 0
NSI = 2
SF_MAX_PAIR_OVERLAP, SF_MAX_PLATE_OVERLAP, SF_MAX_SPRING_SPEED, \
    SF_FRICTION_GAIN = range(4)
NSF = 4
# a collision "gains" energy in its dissipation frame when the gain exceeds
# round-off: this relative tolerance times the frame energy, plus a floor
GAIN_REL_TOL = 1e-10
GAIN_ABS_TOL = 1e-14

INF = np.inf

# fail-fast tolerances: geometric errors larger than these mean a missed or
# mispredicted event, not round-off
PLATE_PENETRATION_TOL = 2e-8    # sphere below the plate surface at prediction
PLATE_CONTACT_TOL = 2e-8        # |gap| at a plate collision
PAIR_PENETRATION_TOL = 2e-8     # pair overlap depth at prediction
PLATE_MAX_SEGMENTS = 1000       # monotone-curvature segments per plate search


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

@njit(cache=True, inline='always')
def restitution(vn_abs, eps, alpha, v0, B):
    if vn_abs >= v0:
        return eps
    return 1.0 - B * vn_abs ** alpha


@njit(cache=True, inline='always')
def advance(i, t, S, g):
    dt = t - S[i, ST]
    if dt != 0.0:
        S[i, SX] += S[i, SVX] * dt
        S[i, SY] += S[i, SVY] * dt
        S[i, SZ] += (S[i, SVZ] - 0.5 * g * dt) * dt
        S[i, SVZ] -= g * dt
        S[i, ST] = t


@njit(cache=True, inline='always')
def cbt_update(i, tree, ev_t, M):
    k = (M + i) >> 1
    while k >= 1:
        a = tree[2 * k]
        b = tree[2 * k + 1]
        if ev_t[a] <= ev_t[b]:
            tree[k] = a
        else:
            tree[k] = b
        k >>= 1


@njit(cache=True)
def cbt_build(tree, ev_t, N, M):
    for k in range(M):
        if k < N:
            tree[M + k] = k
        else:
            tree[M + k] = N      # sentinel, ev_t[N] = +inf
    for k in range(M - 1, 0, -1):
        a = tree[2 * k]
        b = tree[2 * k + 1]
        if ev_t[a] <= ev_t[b]:
            tree[k] = a
        else:
            tree[k] = b


@njit(cache=True, inline='always')
def cell_remove(i, cellof, head, nxt, prv):
    c = cellof[i]
    p = prv[i]
    n = nxt[i]
    if p >= 0:
        nxt[p] = n
    else:
        head[c] = n
    if n >= 0:
        prv[n] = p


@njit(cache=True, inline='always')
def cell_insert(i, c, cellof, head, nxt, prv):
    h = head[c]
    nxt[i] = h
    prv[i] = -1
    if h >= 0:
        prv[h] = i
    head[c] = i
    cellof[i] = c


@njit(cache=True)
def cell_of_position(x, y, z, P, IP):
    nx = IP[I_NX]
    ny = IP[I_NY]
    nz = IP[I_NZ]
    hx = P[P_HX]
    hz = P[P_HZ]
    z0 = P[P_Z0]
    ix = int(x / hx)
    iy = int(y / hx)
    iz = int((z - z0) / hz)
    if ix < 0:
        ix = 0
    if ix > nx - 1:
        ix = nx - 1
    if iy < 0:
        iy = 0
    if iy > ny - 1:
        iy = ny - 1
    if iz < 1:
        iz = 1
    if iz > nz - 1:
        iz = nz - 1
    return ix + nx * (iy + ny * iz)


@njit(cache=True)
def build_cells(S, cellof, head, nxt, prv, P, IP):
    N = IP[I_N]
    for c in range(head.shape[0]):
        head[c] = -1
    for i in range(N - 1, -1, -1):
        cell_insert(i, cell_of_position(S[i, SX], S[i, SY], S[i, SZ], P, IP),
                    cellof, head, nxt, prv)


@njit(cache=True, inline='always')
def particle_energy(i, S, W, mass, K, vz_frame):
    """Kinetic plus rotational energy of particle i in a frame moving
    vertically at vz_frame (I = K m r^2, with K = 4 I / (m sigma^2))."""
    vz = S[i, SVZ] - vz_frame
    v2 = S[i, SVX] * S[i, SVX] + S[i, SVY] * S[i, SVY] + vz * vz
    w2 = W[i, 0] * W[i, 0] + W[i, 1] * W[i, 1] + W[i, 2] * W[i, 2]
    r = S[i, SR]
    return 0.5 * mass[i] * (v2 + K * r * r * w2)


@njit(cache=True, inline='always')
def record_dissipation(slot, before, after, en, cnt):
    """Add a collision's energy change in its dissipation frame to en[slot];
    a gain beyond round-off is counted and its size tracked."""
    change = after - before
    en[slot] += change
    if change > GAIN_REL_TOL * before + GAIN_ABS_TOL:
        cnt[C_EGAIN] += 1
        if change > en[EN_MAX_GAIN]:
            en[EN_MAX_GAIN] = change


# ----------------------------------------------------------------------------
# soft contacts
#
# A soft contact is a spring between two spheres, or a sphere and the plate,
# that replaces hard collisions between them for as long as it lasts.  It is
# created at a touch event (zero compression, velocities untouched) and
# removed at the first kick that finds it open.  Springs are stored once in
# the global list (ca, cb, cgam, cxi); cp[i, :ncp[i]] lists the partners of
# sphere i so prediction can skip them.
# ----------------------------------------------------------------------------

@njit(cache=True, inline='always')
def is_sprung(i, j, cp, ncp):
    for s in range(ncp[i]):
        if cp[i, s] == j:
            return True
    return False


@njit(cache=True)
def partner_add(i, j, cp, ncp):
    if ncp[i] >= MAXC:
        raise RuntimeError("sphere holds too many soft contacts")
    cp[i, ncp[i]] = j
    ncp[i] += 1


@njit(cache=True)
def partner_remove(i, j, cp, ncp):
    for s in range(ncp[i]):
        if cp[i, s] == j:
            ncp[i] -= 1
            cp[i, s] = cp[i, ncp[i]]
            return
    raise RuntimeError("soft-contact partner lists are inconsistent")


@njit(cache=True)
def rebuild_partners(ca, cb, cp, ncp, si):
    for i in range(ncp.shape[0]):
        ncp[i] = 0
    for c in range(si[SI_NC]):
        partner_add(ca[c], cb[c], cp, ncp)
        if cb[c] >= 0:
            partner_add(cb[c], ca[c], cp, ncp)


@njit(cache=True)
def touch_is_soft(i, j, closing, t, last, ncp, P, IP):
    """Whether a touch of sphere i with sphere j (or the plate, j < 0) at
    approach speed closing becomes a spring rather than a hard collision.

    MODE_TC: it does if either sphere collided, or lost a spring, within the
    last P_TCW, or holds a spring now, unless it is faster than P_VSOFT."""
    mode = IP[I_MODE]
    if mode == MODE_NEVER:
        return False
    if mode == MODE_ALWAYS:
        return True
    if closing > P[P_VSOFT]:
        return False
    tcw = P[P_TCW]
    if t - last[i] <= tcw or ncp[i] > 0:
        return True
    if j >= 0 and (t - last[j] <= tcw or ncp[j] > 0):
        return True
    return False


@njit(cache=True)
def spring_add(a, b, closing, meff, t, S, W, ca, cb, cgam, cxi, cwal, cp, ncp,
               si, sf, P, cnt, ev_t, ev_type, tree, IP):
    """Create a spring between sphere a and b (PLATE for the plate), both
    states at time t.  Its damping is fixed here so that an isolated linear
    spring-dashpot collision at this approach speed has the paper's
    restitution e(closing).

    With TANG_WALTON the tangential impulse Walton's operator would deliver
    in this collision is worked out here too (as in collide_pp), to be
    delivered over the collision in proportion to the normal force."""
    c = si[SI_NC]
    if c >= ca.shape[0]:
        raise RuntimeError("soft-contact list is full")
    e = restitution(closing, P[P_EPS], P[P_ALPHA], P[P_V0], P[P_B])
    loge = np.log(e)
    zeta = -loge / np.sqrt(np.pi * np.pi + loge * loge)
    ca[c] = a
    cb[c] = b
    cgam[c] = 2.0 * zeta * np.sqrt(P[P_KN] * meff)
    cxi[c, 0] = 0.0
    cxi[c, 1] = 0.0
    cxi[c, 2] = 0.0
    for q in range(6):
        cwal[c, q] = 0.0
    cwal[c, CW_TIME] = -INF
    if IP[I_TANG] == TANG_WALTON:
        zp = P[P_A] * np.sin(P[P_OM] * t)
        vp = P[P_A] * P[P_OM] * np.cos(P[P_OM] * t)
        nx_, ny_, nz_, delta, vn, vsx, vsy, vsz = contact_state(a, b, S, W, zp, vp)
        vs = np.sqrt(vsx * vsx + vsy * vsy + vsz * vsz)
        Kin = P[P_K]
        if vs > 0.0:
            onepb = P[P_MU] * (1.0 + e) * (1.0 + 1.0 / Kin) * closing / vs
            if onepb > 1.0 + P[P_BETA0]:
                onepb = 1.0 + P[P_BETA0]
            impulse = meff * Kin * onepb / (Kin + 1.0) * vs
            cwal[c, CW_DX] = vsx / vs
            cwal[c, CW_DY] = vsy / vs
            cwal[c, CW_DZ] = vsz / vs
            cwal[c, CW_LEFT] = impulse
            cwal[c, CW_RATIO] = impulse / (meff * (1.0 + e) * closing)
        # how long an isolated collision of this contact lasts
        cwal[c, CW_TIME] = np.pi / np.sqrt(P[P_KN] / meff * (1.0 - zeta * zeta))
    si[SI_NC] = c + 1
    partner_add(a, b, cp, ncp)
    if b >= 0:
        partner_add(b, a, cp, ncp)
    cnt[C_SPRINGS] += 1
    if closing > sf[SF_MAX_SPRING_SPEED]:
        sf[SF_MAX_SPRING_SPEED] = closing
    N = IP[I_N]
    if ev_t[N] == INF:
        ev_t[N] = t + P[P_DT]
        ev_type[N] = EV_KICK
        cbt_update(N, tree, ev_t, IP[I_M])


@njit(cache=True)
def spring_energy(t, S, ca, cb, cxi, si, P):
    """Elastic energy stored in the springs, with states projected to t."""
    g = P[P_G]
    kn = P[P_KN]
    kt = P[P_KT]
    total = 0.0
    for c in range(si[SI_NC]):
        a = ca[c]
        b = cb[c]
        dta = t - S[a, ST]
        za = S[a, SZ] + (S[a, SVZ] - 0.5 * g * dta) * dta
        if b >= 0:
            dtb = t - S[b, ST]
            dx = S[b, SX] + S[b, SVX] * dtb - S[a, SX] - S[a, SVX] * dta
            dy = S[b, SY] + S[b, SVY] * dtb - S[a, SY] - S[a, SVY] * dta
            dz = S[b, SZ] + (S[b, SVZ] - 0.5 * g * dtb) * dtb - za
            delta = S[a, SR] + S[b, SR] - np.sqrt(dx * dx + dy * dy + dz * dz)
        else:
            delta = P[P_A] * np.sin(P[P_OM] * t) + S[a, SR] - za
        if delta > 0.0:
            total += 0.5 * kn * delta * delta
        total += 0.5 * kt * (cxi[c, 0] ** 2 + cxi[c, 1] ** 2 + cxi[c, 2] ** 2)
    return total


# ----------------------------------------------------------------------------
# collision operators (Walton 1993, section 25.5)
# ----------------------------------------------------------------------------

@njit(cache=True)
def collide_pp(i, j, S, W, mass, P, cnt):
    """Binary collision of spheres i (=a) and j (=b), states at the same time."""
    eps = P[P_EPS]
    K = P[P_K]
    mu = P[P_MU]
    beta0 = P[P_BETA0]
    dx = S[j, SX] - S[i, SX]
    dy = S[j, SY] - S[i, SY]
    dz = S[j, SZ] - S[i, SZ]
    dist = np.sqrt(dx * dx + dy * dy + dz * dz)
    nx_ = dx / dist
    ny_ = dy / dist
    nz_ = dz / dist
    # relative velocity v_ab = v_b - v_a
    vax = S[j, SVX] - S[i, SVX]
    vay = S[j, SVY] - S[i, SVY]
    vaz = S[j, SVZ] - S[i, SVZ]
    vn = vax * nx_ + vay * ny_ + vaz * nz_
    if vn >= 0.0:
        cnt[C_PPSKIP] += 1
        return False
    ri = S[i, SR]
    rj = S[j, SR]
    # tangential relative velocity and relative surface velocity
    vtx = vax - vn * nx_
    vty = vay - vn * ny_
    vtz = vaz - vn * nz_
    wsx = ri * W[i, 0] + rj * W[j, 0]
    wsy = ri * W[i, 1] + rj * W[j, 1]
    wsz = ri * W[i, 2] + rj * W[j, 2]
    # v_s = v_t + n x (r_a w_a + r_b w_b)
    vsx = vtx + (ny_ * wsz - nz_ * wsy)
    vsy = vty + (nz_ * wsx - nx_ * wsz)
    vsz = vtz + (nx_ * wsy - ny_ * wsx)
    e = restitution(-vn, eps, P[P_ALPHA], P[P_V0], P[P_B])
    if -vn < 1e-10:
        cnt[C_TINYPP] += 1
    mi = mass[i]
    mj = mass[j]
    meff = mi * mj / (mi + mj)
    vs2 = vsx * vsx + vsy * vsy + vsz * vsz
    onepb = 0.0
    if vs2 > 0.0:
        # sliding solution (Walton Eq. 25.23), capped by the rolling limit beta0
        onepb = mu * (1.0 + e) * (1.0 + 1.0 / K) * (-vn) / np.sqrt(vs2)
        if onepb > 1.0 + beta0:
            onepb = 1.0 + beta0
    pn = meff * (1.0 + e) * vn             # normal impulse on a (along n)
    ft = meff * K * onepb / (K + 1.0)       # tangential impulse on a = ft * vs
    ptx = ft * vsx
    pty = ft * vsy
    ptz = ft * vsz
    px = pn * nx_ + ptx
    py = pn * ny_ + pty
    pz = pn * nz_ + ptz
    S[i, SVX] += px / mi
    S[i, SVY] += py / mi
    S[i, SVZ] += pz / mi
    S[j, SVX] -= px / mj
    S[j, SVY] -= py / mj
    S[j, SVZ] -= pz / mj
    # n x P_t
    cx = ny_ * ptz - nz_ * pty
    cy = nz_ * ptx - nx_ * ptz
    cz = nx_ * pty - ny_ * ptx
    fi = 1.0 / (K * mi * ri)     # 2/(K m sigma) with sigma = 2 r
    fj = 1.0 / (K * mj * rj)
    W[i, 0] += fi * cx
    W[i, 1] += fi * cy
    W[i, 2] += fi * cz
    W[j, 0] += fj * cx
    W[j, 1] += fj * cy
    W[j, 2] += fj * cz
    return True


@njit(cache=True)
def collide_wall(i, nx_, ny_, nz_, wvx, wvy, wvz, S, W, P):
    """Collision of sphere i with an infinitely massive flat wall.

    (nx_, ny_, nz_) is the unit normal pointing from the sphere into the wall,
    (wvx, wvy, wvz) the wall velocity.  Returns the post-collision normal
    separation speed, or -1 if the sphere was not approaching.
    """
    eps = P[P_EPS]
    K = P[P_K]
    mu = P[P_MU]
    beta0 = P[P_BETA0]
    vax = wvx - S[i, SVX]
    vay = wvy - S[i, SVY]
    vaz = wvz - S[i, SVZ]
    vn = vax * nx_ + vay * ny_ + vaz * nz_
    if vn >= 0.0:
        return -1.0
    ri = S[i, SR]
    vtx = vax - vn * nx_
    vty = vay - vn * ny_
    vtz = vaz - vn * nz_
    wsx = ri * W[i, 0]
    wsy = ri * W[i, 1]
    wsz = ri * W[i, 2]
    vsx = vtx + (ny_ * wsz - nz_ * wsy)
    vsy = vty + (nz_ * wsx - nx_ * wsz)
    vsz = vtz + (nx_ * wsy - ny_ * wsx)
    e = restitution(-vn, eps, P[P_ALPHA], P[P_V0], P[P_B])
    vs2 = vsx * vsx + vsy * vsy + vsz * vsz
    onepb = 0.0
    if vs2 > 0.0:
        onepb = mu * (1.0 + e) * (1.0 + 1.0 / K) * (-vn) / np.sqrt(vs2)
        if onepb > 1.0 + beta0:
            onepb = 1.0 + beta0
    # impulse per unit mass on the sphere (m_eff = m for an immovable wall)
    pn = (1.0 + e) * vn
    ft = K * onepb / (K + 1.0)
    ptx = ft * vsx
    pty = ft * vsy
    ptz = ft * vsz
    S[i, SVX] += pn * nx_ + ptx
    S[i, SVY] += pn * ny_ + pty
    S[i, SVZ] += pn * nz_ + ptz
    cx = ny_ * ptz - nz_ * pty
    cy = nz_ * ptx - nx_ * ptz
    cz = nx_ * pty - ny_ * ptx
    fi = 1.0 / (K * ri)
    W[i, 0] += fi * cx
    W[i, 1] += fi * cy
    W[i, 2] += fi * cz
    return -e * vn


# ----------------------------------------------------------------------------
# event prediction
# ----------------------------------------------------------------------------

@njit(cache=True, inline='always')
def plate_gap(tau, gap0, vz, phase, A, om, g):
    """Gap between sphere bottom and plate at t + tau, given the gap gap0 at t.

    The plate displacement A sin(phase + om tau) - A sin(phase) is written as
    2 A cos(phase + om tau / 2) sin(om tau / 2), which avoids cancellation
    when tau is small.
    """
    return (gap0 + vz * tau - 0.5 * g * tau * tau
            - 2.0 * A * np.cos(phase + 0.5 * om * tau) * np.sin(0.5 * om * tau))


@njit(cache=True, inline='always')
def plate_gap_rate(tau, vz, phase, A, om, g):
    return vz - g * tau - A * om * np.cos(phase + om * tau)


@njit(cache=True)
def plate_gap_root(lower, upper, gap0, vz, phase, A, om, g):
    """Root of the gap in [lower, upper] with gap(lower) > 0 >= gap(upper):
    Newton steps safeguarded by bisection."""
    trial = upper
    for _ in range(100):
        value = plate_gap(trial, gap0, vz, phase, A, om, g)
        if value > 0.0:
            lower = trial
        else:
            upper = trial
        rate = plate_gap_rate(trial, vz, phase, A, om, g)
        if rate != 0.0 and abs(value / rate) < 1e-14 * max(1.0, trial):
            return trial
        if upper - lower < 2e-14 * max(1.0, upper):
            return 0.5 * (lower + upper)
        if rate != 0.0:
            candidate = trial - value / rate
        else:
            candidate = lower
        if candidate <= lower or candidate >= upper:
            candidate = 0.5 * (lower + upper)
        trial = candidate
    return 0.5 * (lower + upper)


@njit(cache=True)
def plate_root(zi, vzi, ri, t, horizon, A, om, g):
    """First time tau in [0, horizon] at which the sphere (state at time t)
    touches the plate z_p = A sin(om t') while approaching it, or INF.

    The gap h(tau) = zi + vzi tau - g tau^2/2 - ri - A sin(om (t + tau)) has
    curvature -g + A om^2 sin(om (t + tau)), which changes sign only where
    sin(om (t + tau)) = g / (A om^2).  Between those points h is convex or
    concave, so it has at most one extremum; that extremum is located from the
    sign of h', and the first sign change of h is then bracketed and solved.
    Raises if the sphere starts inside the plate.
    """
    phase = (om * t) % (2.0 * np.pi)
    gap0 = zi - ri - A * np.sin(phase)
    if gap0 < -PLATE_PENETRATION_TOL:
        raise ValueError("sphere penetrates the bottom plate")
    if gap0 < 0.0:
        gap0 = 0.0
    curv_max = g + A * om * om
    # lower bound of h on [0, horizon] from h'(0) >= vzi - A om, |h''| <= curv_max
    if gap0 + min(0.0, vzi - A * om) * horizon - 0.5 * curv_max * horizon * horizon > 1e-12:
        return INF
    if gap0 == 0.0 and plate_gap_rate(0.0, vzi, phase, A, om, g) < 0.0:
        return 0.0
    # the sphere bottom stays above the highest plate position
    if min(zi, zi + vzi * horizon - 0.5 * g * horizon * horizon) - ri > A:
        return INF
    split = A * om * om > g
    turn = 0.0
    if split:
        turn = np.arcsin(g / (A * om * om))
    two_pi = 2.0 * np.pi
    lower = 0.0
    for _ in range(PLATE_MAX_SEGMENTS):
        upper = horizon
        if split:
            psi = phase + om * lower
            turn1 = turn + two_pi * (np.floor((psi - turn) / two_pi) + 1.0)
            turn2 = np.pi - turn + two_pi * (np.floor((psi - np.pi + turn) / two_pi) + 1.0)
            upper = min(horizon, (min(turn1, turn2) - phase) / om)
            if upper <= lower + 1e-14:
                upper = min(horizon, lower + 1e-12)
        rate_lower = plate_gap_rate(lower, vzi, phase, A, om, g)
        rate_upper = plate_gap_rate(upper, vzi, phase, A, om, g)
        middle = upper
        if rate_lower * rate_upper < 0.0:
            # isolate the extremum of h in this segment
            left = lower
            right = upper
            middle = 0.5 * (left + right)
            for _ in range(100):
                rate_middle = plate_gap_rate(middle, vzi, phase, A, om, g)
                if rate_middle * rate_lower > 0.0:
                    left = middle
                else:
                    right = middle
                curvature = -g + A * om * om * np.sin(phase + om * middle)
                if curvature != 0.0 and abs(rate_middle / curvature) < 1e-14:
                    break
                if right - left < 2e-14:
                    middle = 0.5 * (left + right)
                    break
                if curvature != 0.0:
                    candidate = middle - rate_middle / curvature
                else:
                    candidate = left
                if candidate <= left or candidate >= right:
                    candidate = 0.5 * (left + right)
                middle = candidate
        if plate_gap(middle, gap0, vzi, phase, A, om, g) < 0.0:
            return plate_gap_root(lower, middle, gap0, vzi, phase, A, om, g)
        if plate_gap(upper, gap0, vzi, phase, A, om, g) < 0.0:
            return plate_gap_root(middle, upper, gap0, vzi, phase, A, om, g)
        if upper >= horizon:
            return INF
        lower = upper
    raise RuntimeError("plate root isolation exceeded the segment limit")


@njit(cache=True)
def predict(i, t, S, cellof, head, nxt, ev_t, P, IP, cnt, cp, ncp):
    """Earliest event of particle i (state at time t) given the current
    states of its neighbours.  Pair events are only accepted if they occur
    before the partner's own scheduled event (Lubachevsky invariant).
    Partners i holds a spring with (spheres or the plate) are skipped."""
    L = P[P_L]
    hx = P[P_HX]
    hz = P[P_HZ]
    z0 = P[P_Z0]
    g = P[P_G]
    btol = P[P_BTOL]
    nx = IP[I_NX]
    ny = IP[I_NY]
    nz = IP[I_NZ]
    xi = S[i, SX]
    yi = S[i, SY]
    zi = S[i, SZ]
    vxi = S[i, SVX]
    vyi = S[i, SVY]
    vzi = S[i, SVZ]
    ri = S[i, SR]
    c = cellof[i]
    ix = c % nx
    tmp = c // nx
    iy = tmp % ny
    iz = tmp // ny

    best = INF
    btype = EV_ADV
    bp = -1
    # --- x: cell faces / side walls
    if vxi > 0.0:
        if ix < nx - 1:
            d = (ix + 1) * hx - xi
            if d < 0.0:
                d = 0.0
            tt = d / vxi
            if tt < best:
                best = tt
                btype = EV_CELL
                bp = 1
        else:
            d = L - ri - xi
            if d < 0.0:
                d = 0.0
            tt = d / vxi
            if tt < best:
                best = tt
                btype = EV_WALL
                bp = 1
    elif vxi < 0.0:
        if ix > 0:
            d = xi - ix * hx
            if d < 0.0:
                d = 0.0
            tt = d / (-vxi)
            if tt < best:
                best = tt
                btype = EV_CELL
                bp = 0
        else:
            d = xi - ri
            if d < 0.0:
                d = 0.0
            tt = d / (-vxi)
            if tt < best:
                best = tt
                btype = EV_WALL
                bp = 0
    # --- y
    if vyi > 0.0:
        if iy < ny - 1:
            d = (iy + 1) * hx - yi
            if d < 0.0:
                d = 0.0
            tt = d / vyi
            if tt < best:
                best = tt
                btype = EV_CELL
                bp = 3
        else:
            d = L - ri - yi
            if d < 0.0:
                d = 0.0
            tt = d / vyi
            if tt < best:
                best = tt
                btype = EV_WALL
                bp = 3
    elif vyi < 0.0:
        if iy > 0:
            d = yi - iy * hx
            if d < 0.0:
                d = 0.0
            tt = d / (-vyi)
            if tt < best:
                best = tt
                btype = EV_CELL
                bp = 2
        else:
            d = yi - ri
            if d < 0.0:
                d = 0.0
            tt = d / (-vyi)
            if tt < best:
                best = tt
                btype = EV_WALL
                bp = 2
    # --- z faces (parabolic motion)
    if iz < nz - 1 and vzi > 0.0:
        d = z0 + (iz + 1) * hz - zi
        if d < 0.0:
            d = 0.0
        disc = vzi * vzi - 2.0 * g * d
        if disc >= 0.0:
            tt = 2.0 * d / (vzi + np.sqrt(disc))
            if tt < best:
                best = tt
                btype = EV_CELL
                bp = 5
    if iz > 0:
        d = zi - (z0 + iz * hz)
        if d < 0.0:
            d = 0.0
        S_ = np.sqrt(vzi * vzi + 2.0 * g * d)
        if vzi > 0.0:
            tt = (vzi + S_) / g
        else:
            den = S_ - vzi
            if den > 0.0:
                tt = 2.0 * d / den
            else:
                tt = 0.0
        if tt < best:
            best = tt
            btype = EV_CELL
            bp = 4

    # --- particle pairs in the 27 neighbouring cells
    npair = 0
    nsprung = ncp[i]
    for jz in range(iz - 1, iz + 2):
        if jz < 0 or jz >= nz:
            continue
        for jy in range(iy - 1, iy + 2):
            if jy < 0 or jy >= ny:
                continue
            base = nx * (jy + ny * jz)
            for jx in range(ix - 1, ix + 2):
                if jx < 0 or jx >= nx:
                    continue
                j = head[base + jx]
                while j >= 0:
                    npair += 1
                    if j != i and (nsprung == 0 or not is_sprung(i, j, cp, ncp)):
                        dtj = t - S[j, ST]
                        dvz = S[j, SVZ] - g * dtj - vzi
                        ddx = S[j, SX] + S[j, SVX] * dtj - xi
                        ddy = S[j, SY] + S[j, SVY] * dtj - yi
                        ddz = S[j, SZ] + (S[j, SVZ] - 0.5 * g * dtj) * dtj - zi
                        dvx = S[j, SVX] - vxi
                        dvy = S[j, SVY] - vyi
                        b = ddx * dvx + ddy * dvy + ddz * dvz
                        if b < 0.0:
                            sig = ri + S[j, SR]
                            cc = ddx * ddx + ddy * ddy + ddz * ddz - sig * sig
                            a = dvx * dvx + dvy * dvy + dvz * dvz
                            if cc <= 0.0:
                                # touching / overlapping by round-off
                                if cc < -2.0 * sig * PAIR_PENETRATION_TOL:
                                    raise ValueError("sphere pair overlaps")
                                if b < -btol * np.sqrt(a):
                                    tt = 0.0
                                else:
                                    tt = INF
                            else:
                                disc = b * b - a * cc
                                if disc > 0.0:
                                    tt = cc / (np.sqrt(disc) - b)
                                else:
                                    tt = INF
                            if tt < best and t + tt < ev_t[j]:
                                best = tt
                                btype = EV_PP
                                bp = j
                    j = nxt[j]

    cnt[C_PAIRS] += npair
    cnt[C_PREDICT] += 1
    # --- bottom plate (only reachable from low cells)
    A = P[P_A]
    if z0 + iz * hz - ri <= A and (nsprung == 0
                                   or not is_sprung(i, PLATE, cp, ncp)):
        if best == INF:
            raise RuntimeError("plate search without a finite horizon")
        tpl = plate_root(zi, vzi, ri, t, best, A, P[P_OM], g)
        cnt[C_ROOTCALL] += 1
        if tpl < best:
            best = tpl
            btype = EV_PLATE
            bp = 0
    if best == INF:
        best = 1.0
        btype = EV_ADV
        bp = -1
    return t + best, btype, bp


@njit(cache=True)
def schedule(i, t, S, cellof, head, nxt, ev_t, ev_type, ev_p, tree, P, IP,
             cnt, cp, ncp):
    tb, typ, p = predict(i, t, S, cellof, head, nxt, ev_t, P, IP, cnt, cp, ncp)
    M = IP[I_M]
    if typ == EV_PP:
        j = p
        if ev_type[j] == EV_PP:
            m = ev_p[j]
            if m != i:
                ev_type[m] = EV_ADV     # preempted: m re-predicts at its time
        ev_t[j] = tb
        ev_type[j] = EV_PP
        ev_p[j] = i
        cbt_update(j, tree, ev_t, M)
    ev_t[i] = tb
    ev_type[i] = typ
    ev_p[i] = p
    cbt_update(i, tree, ev_t, M)


# ----------------------------------------------------------------------------
# kick: spring impulses
# ----------------------------------------------------------------------------

@njit(cache=True, inline='always')
def contact_state(a, b, S, W, zp, vp):
    """Geometry and relative motion of the contact of sphere a with sphere b
    (or the plate, b < 0), both states at the same time: unit normal from a
    into b, compression, normal relative velocity (negative when closing),
    and the relative surface velocity as in collide_pp."""
    ra = S[a, SR]
    if b >= 0:
        rb = S[b, SR]
        dx = S[b, SX] - S[a, SX]
        dy = S[b, SY] - S[a, SY]
        dz = S[b, SZ] - S[a, SZ]
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        nx_ = dx / dist
        ny_ = dy / dist
        nz_ = dz / dist
        delta = ra + rb - dist
        ux = S[b, SVX] - S[a, SVX]
        uy = S[b, SVY] - S[a, SVY]
        uz = S[b, SVZ] - S[a, SVZ]
        wsx = ra * W[a, 0] + rb * W[b, 0]
        wsy = ra * W[a, 1] + rb * W[b, 1]
        wsz = ra * W[a, 2] + rb * W[b, 2]
    else:
        nx_ = 0.0
        ny_ = 0.0
        nz_ = -1.0
        delta = zp + ra - S[a, SZ]
        ux = -S[a, SVX]
        uy = -S[a, SVY]
        uz = vp - S[a, SVZ]
        wsx = ra * W[a, 0]
        wsy = ra * W[a, 1]
        wsz = ra * W[a, 2]
    vn = ux * nx_ + uy * ny_ + uz * nz_
    vsx = ux - vn * nx_ + (ny_ * wsz - nz_ * wsy)
    vsy = uy - vn * ny_ + (nz_ * wsx - nx_ * wsz)
    vsz = uz - vn * nz_ + (nx_ * wsy - ny_ * wsx)
    return nx_, ny_, nz_, delta, vn, vsx, vsy, vsz


@njit(cache=True)
def kick(t, S, W, mass, cellof, head, nxt, ev_t, ev_type, ev_p, tree, P, IP,
         cnt, en, ca, cb, cgam, cxi, cwal, cscr, cp, ncp, last, dV, dW, flag,
         touched, si, sf):
    """Apply one time step's impulse of every spring at time t.

    Normal force on a contact compressed by delta and closing at rate c:
    kn delta + gamma c, along the line of centres (not clamped at zero, so an
    isolated collision has exactly the restitution its damping was set for).
    Tangential force, sustained phase: kt xi, where xi accumulates the
    relative surface displacement, shortened so that
    kt |xi| <= mu max(normal force, 0) (Coulomb sliding).  With TANG_WALTON
    a spring first goes through a collision phase, for as long as an isolated
    collision of that contact lasts (plus one kick): the tangential impulse
    is the Walton impulse worked out at creation, delivered as its ratio to
    the normal impulse times the normal force, up to its total.  An isolated
    collision therefore ends with Walton's tangential result, and only a
    contact that outlasts one gets the spring (from zero stretch).  All forces are
    computed from the same state, then applied, then every sphere that was
    kicked re-predicts its events.  A spring found open is removed.

    Energy books: EN_SOFT_DISS gets the dashpot's work (its impulse times the
    mean closing rate across the kick), the friction work of sliding
    contacts, and the tangential spring energy dropped when a spring is
    removed; plate springs add to EN_PLATE_WORK and EN_PLATE_JZ like plate
    collisions.  What is left over in the audit is
    the integrator's error."""
    g = P[P_G]
    kn = P[P_KN]
    kt = P[P_KT]
    dt = P[P_DT]
    mu = P[P_MU]
    Kin = P[P_K]
    zp = P[P_A] * np.sin(P[P_OM] * t)
    vp = P[P_A] * P[P_OM] * np.cos(P[P_OM] * t)
    ntouch = 0
    c = 0
    while c < si[SI_NC]:
        a = ca[c]
        b = cb[c]
        advance(a, t, S, g)
        ra = S[a, SR]
        if b >= 0:
            advance(b, t, S, g)
            rb = S[b, SR]
        nx_, ny_, nz_, delta, vn, vsx, vsy, vsz = contact_state(a, b, S, W, zp, vp)
        xx = cxi[c, 0]
        xy = cxi[c, 1]
        xz = cxi[c, 2]
        if not flag[a]:
            flag[a] = 1
            touched[ntouch] = a
            ntouch += 1
        if b >= 0 and not flag[b]:
            flag[b] = 1
            touched[ntouch] = b
            ntouch += 1
        if delta <= 0.0:
            en[EN_SOFT_DISS] -= 0.5 * kt * (xx * xx + xy * xy + xz * xz)
            last[a] = t
            partner_remove(a, b, cp, ncp)
            if b >= 0:
                last[b] = t
                partner_remove(b, a, cp, ncp)
            m = si[SI_NC] - 1
            ca[c] = ca[m]
            cb[c] = cb[m]
            cgam[c] = cgam[m]
            cxi[c, 0] = cxi[m, 0]
            cxi[c, 1] = cxi[m, 1]
            cxi[c, 2] = cxi[m, 2]
            for q in range(6):
                cwal[c, q] = cwal[m, q]
            si[SI_NC] = m
            continue
        closing = -vn
        fn = kn * delta + cgam[c] * closing
        cscr[c, 2] = 0.0
        if cwal[c, CW_TIME] > -dt:
            # collision phase: deliver the Walton impulse with the normal force
            cwal[c, CW_TIME] -= dt
            ex = cwal[c, CW_DX]
            ey = cwal[c, CW_DY]
            ez = cwal[c, CW_DZ]
            along = ex * nx_ + ey * ny_ + ez * nz_
            ex -= along * nx_
            ey -= along * ny_
            ez -= along * nz_
            flat = np.sqrt(ex * ex + ey * ey + ez * ez)
            size = 0.0
            if flat > 0.0:
                ex /= flat
                ey /= flat
                ez /= flat
                size = min(cwal[c, CW_RATIO] * max(fn, 0.0) * dt, cwal[c, CW_LEFT])
                cwal[c, CW_DX] = ex
                cwal[c, CW_DY] = ey
                cwal[c, CW_DZ] = ez
                cwal[c, CW_LEFT] -= size
            jtx = size * ex
            jty = size * ey
            jtz = size * ez
            if size > 0.0:
                cscr[c, 2] = 1.0
        else:
            # sustained phase: turn xi into the current tangent plane, keeping
            # its length, then stretch it by the surface displacement
            length = np.sqrt(xx * xx + xy * xy + xz * xz)
            along = xx * nx_ + xy * ny_ + xz * nz_
            xx -= along * nx_
            xy -= along * ny_
            xz -= along * nz_
            flat = np.sqrt(xx * xx + xy * xy + xz * xz)
            if flat > 0.0:
                scale = length / flat
                xx *= scale
                xy *= scale
                xz *= scale
            stored = length * length
            xx += vsx * dt
            xy += vsy * dt
            xz += vsz * dt
            cap = mu * max(fn, 0.0)
            length = np.sqrt(xx * xx + xy * xy + xz * xz)
            if kt * length > cap:
                # sliding: the spring is cut back to the Coulomb limit.  The
                # energy lost is the friction force's work on the slip
                # (booked after the kick) plus the change in the spring's
                # stored energy.
                shorter = cap / kt
                scale = shorter / length
                xx *= scale
                xy *= scale
                xz *= scale
                en[EN_SOFT_DISS] += 0.5 * kt * (shorter * shorter - stored)
                cscr[c, 2] = 1.0
            cxi[c, 0] = xx
            cxi[c, 1] = xy
            cxi[c, 2] = xz
            jtx = kt * xx * dt
            jty = kt * xy * dt
            jtz = kt * xz * dt
        # friction work is booked after the kick from the mean surface velocity
        cscr[c, 3] = vsx
        cscr[c, 4] = vsy
        cscr[c, 5] = vsz
        cscr[c, 6] = jtx
        cscr[c, 7] = jty
        cscr[c, 8] = jtz
        # impulse on a: normal pushes it away from b, friction drags it along
        # b's surface motion
        jx = -fn * dt * nx_ + jtx
        jy = -fn * dt * ny_ + jty
        jz = -fn * dt * nz_ + jtz
        tx = ny_ * jtz - nz_ * jty
        ty = nz_ * jtx - nx_ * jtz
        tz = nx_ * jty - ny_ * jtx
        ma = mass[a]
        dV[a, 0] += jx / ma
        dV[a, 1] += jy / ma
        dV[a, 2] += jz / ma
        fa = 1.0 / (Kin * ma * ra)
        dW[a, 0] += fa * tx
        dW[a, 1] += fa * ty
        dW[a, 2] += fa * tz
        if b >= 0:
            mb = mass[b]
            dV[b, 0] -= jx / mb
            dV[b, 1] -= jy / mb
            dV[b, 2] -= jz / mb
            fb = 1.0 / (Kin * mb * rb)
            dW[b, 0] += fb * tx
            dW[b, 1] += fb * ty
            dW[b, 2] += fb * tz
            if delta > sf[SF_MAX_PAIR_OVERLAP]:
                sf[SF_MAX_PAIR_OVERLAP] = delta
        else:
            en[EN_PLATE_JZ] += jz
            en[EN_PLATE_WORK] += vp * jz
            if delta > sf[SF_MAX_PLATE_OVERLAP]:
                sf[SF_MAX_PLATE_OVERLAP] = delta
        cscr[c, 0] = cgam[c] * closing * dt
        cscr[c, 1] = closing
        cnt[C_KICKF] += 1
        c += 1
    for k in range(ntouch):
        i = touched[k]
        S[i, SVX] += dV[i, 0]
        S[i, SVY] += dV[i, 1]
        S[i, SVZ] += dV[i, 2]
        W[i, 0] += dW[i, 0]
        W[i, 1] += dW[i, 1]
        W[i, 2] += dW[i, 2]
        dV[i, 0] = 0.0
        dV[i, 1] = 0.0
        dV[i, 2] = 0.0
        dW[i, 0] = 0.0
        dW[i, 1] = 0.0
        dW[i, 2] = 0.0
        flag[i] = 0
    # dashpot work: its impulse times the mean closing rate across the kick;
    # friction work of sliding contacts: likewise with the surface velocity
    for c in range(si[SI_NC]):
        nx_, ny_, nz_, delta, vn, vsx, vsy, vsz = contact_state(
            ca[c], cb[c], S, W, zp, vp)
        en[EN_SOFT_DISS] -= cscr[c, 0] * 0.5 * (cscr[c, 1] - vn)
        if cscr[c, 2] != 0.0:
            work = -0.5 * (cscr[c, 6] * (cscr[c, 3] + vsx)
                           + cscr[c, 7] * (cscr[c, 4] + vsy)
                           + cscr[c, 8] * (cscr[c, 5] + vsz))
            en[EN_SOFT_DISS] += work
            if work > 0.0:
                sf[SF_FRICTION_GAIN] += work
    # every kicked sphere re-predicts; an event it was the partner of is void
    for k in range(ntouch):
        i = touched[k]
        if ev_type[i] == EV_PP:
            m = ev_p[i]
            if ev_type[m] == EV_PP and ev_p[m] == i:
                ev_type[m] = EV_ADV
        schedule(i, t, S, cellof, head, nxt, ev_t, ev_type, ev_p, tree, P, IP,
                 cnt, cp, ncp)
    N = IP[I_N]
    if si[SI_NC] > 0:
        ev_t[N] = t + dt
    else:
        ev_t[N] = INF
    cbt_update(N, tree, ev_t, IP[I_M])
    cnt[C_KICKS] += 1


# ----------------------------------------------------------------------------
# main loop
# ----------------------------------------------------------------------------

@njit(cache=True)
def run_events(S, W, mass, cellof, head, nxt, prv, ev_t, ev_type, ev_p, tree,
               P, IP, t_stop, max_events, cnt, en, ca, cb, cgam, cxi, cwal, cscr,
               cp, ncp, last, dV, dW, flag, touched, si, sf):
    """Process all events with time < t_stop (or until max_events).
    Returns 0 if t_stop was reached, 1 if max_events was hit.

    Energy accounting in en (the referee's energy audit uses it): for every
    collision, the energy change in the frame where it can only dissipate
    (the pair's centre of mass, the static wall, the moving plate) goes to
    EN_PP_DISS, EN_WALL_DISS or EN_PLATE_DISS; the plate's work, its
    velocity times the vertical impulse it delivers, to EN_PLATE_WORK; the
    vertical impulses of plate and walls to EN_PLATE_JZ and EN_WALL_JZ.
    Between events energy changes only by gravity, which the audit counts
    through potential energy.  So the layer's energy change equals plate
    work plus the three dissipations, and any other change is energy the
    contact handling put in or took out unaccounted.  A contact method must
    route every velocity or position change it makes through these terms."""
    g = P[P_G]
    Kin = P[P_K]
    A = P[P_A]
    om = P[P_OM]
    nx = IP[I_NX]
    ny = IP[I_NY]
    nev = 0
    while True:
        i = tree[1]
        t = ev_t[i]
        if t >= t_stop:
            return 0
        if nev >= max_events:
            return 1
        nev += 1
        cnt[C_EVENTS] += 1
        typ = ev_type[i]
        if typ == EV_KICK:
            kick(t, S, W, mass, cellof, head, nxt, ev_t, ev_type, ev_p, tree,
                 P, IP, cnt, en, ca, cb, cgam, cxi, cwal, cscr, cp, ncp, last,
                 dV, dW, flag, touched, si, sf)
        elif typ == EV_PP:
            j = ev_p[i]
            advance(i, t, S, g)
            advance(j, t, S, g)
            closing = 0.0
            if IP[I_MODE] != MODE_NEVER:
                dx = S[j, SX] - S[i, SX]
                dy = S[j, SY] - S[i, SY]
                dz = S[j, SZ] - S[i, SZ]
                closing = -((S[j, SVX] - S[i, SVX]) * dx
                            + (S[j, SVY] - S[i, SVY]) * dy
                            + (S[j, SVZ] - S[i, SVZ]) * dz) / np.sqrt(
                                dx * dx + dy * dy + dz * dz)
            if closing > 0.0 and touch_is_soft(i, j, closing, t, last, ncp, P, IP):
                spring_add(i, j, closing,
                           mass[i] * mass[j] / (mass[i] + mass[j]), t, S, W,
                           ca, cb, cgam, cxi, cwal, cp, ncp, si, sf, P, cnt,
                           ev_t, ev_type, tree, IP)
            else:
                before = (particle_energy(i, S, W, mass, Kin, 0.0)
                          + particle_energy(j, S, W, mass, Kin, 0.0))
                if collide_pp(i, j, S, W, mass, P, cnt):
                    cnt[C_PP] += 1
                    last[i] = t
                    last[j] = t
                # momentum is conserved, so the lab-frame change equals the
                # centre-of-mass-frame change, which can only be a loss
                record_dissipation(EN_PP_DISS, before,
                                   particle_energy(i, S, W, mass, Kin, 0.0)
                                   + particle_energy(j, S, W, mass, Kin, 0.0),
                                   en, cnt)
            ev_type[j] = EV_ADV
            ev_t[j] = t
            schedule(i, t, S, cellof, head, nxt, ev_t, ev_type, ev_p, tree, P,
                     IP, cnt, cp, ncp)
            schedule(j, t, S, cellof, head, nxt, ev_t, ev_type, ev_p, tree, P,
                     IP, cnt, cp, ncp)
        elif typ == EV_CELL:
            advance(i, t, S, g)
            f = ev_p[i]
            c = cellof[i]
            if f == 0:
                cn = c - 1
            elif f == 1:
                cn = c + 1
            elif f == 2:
                cn = c - nx
            elif f == 3:
                cn = c + nx
            elif f == 4:
                cn = c - nx * ny
            else:
                cn = c + nx * ny
            cell_remove(i, cellof, head, nxt, prv)
            cell_insert(i, cn, cellof, head, nxt, prv)
            cnt[C_CELL] += 1
            schedule(i, t, S, cellof, head, nxt, ev_t, ev_type, ev_p, tree, P,
                     IP, cnt, cp, ncp)
        elif typ == EV_PLATE:
            advance(i, t, S, g)
            if abs(S[i, SZ] - S[i, SR] - A * np.sin(om * t)) > PLATE_CONTACT_TOL:
                raise ValueError("plate collision off the plate surface")
            vp = A * om * np.cos(om * t)
            closing = vp - S[i, SVZ]
            if closing > 0.0 and touch_is_soft(i, PLATE, closing, t, last, ncp,
                                               P, IP):
                spring_add(i, PLATE, closing, mass[i], t, S, W, ca, cb, cgam,
                           cxi, cwal, cp, ncp, si, sf, P, cnt, ev_t, ev_type,
                           tree, IP)
            else:
                vz_before = S[i, SVZ]
                before = particle_energy(i, S, W, mass, Kin, vp)
                vsep = collide_wall(i, 0.0, 0.0, -1.0, 0.0, 0.0, vp, S, W, P)
                jz = mass[i] * (S[i, SVZ] - vz_before)
                en[EN_PLATE_JZ] += jz
                en[EN_PLATE_WORK] += vp * jz
                record_dissipation(EN_PLATE_DISS, before,
                                   particle_energy(i, S, W, mass, Kin, vp), en,
                                   cnt)
                if vsep >= 0.0:
                    cnt[C_PLATE] += 1
                    last[i] = t
                    if vsep < 1e-10:
                        cnt[C_TINYPL] += 1
                else:
                    cnt[C_PLSKIP] += 1
            schedule(i, t, S, cellof, head, nxt, ev_t, ev_type, ev_p, tree, P,
                     IP, cnt, cp, ncp)
        elif typ == EV_WALL:
            advance(i, t, S, g)
            f = ev_p[i]
            vz_before = S[i, SVZ]
            before = particle_energy(i, S, W, mass, Kin, 0.0)
            if f == 0:
                vs = collide_wall(i, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, S, W, P)
            elif f == 1:
                vs = collide_wall(i, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, S, W, P)
            elif f == 2:
                vs = collide_wall(i, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, S, W, P)
            else:
                vs = collide_wall(i, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, S, W, P)
            if vs >= 0.0:
                cnt[C_WALL] += 1
                last[i] = t
            en[EN_WALL_JZ] += mass[i] * (S[i, SVZ] - vz_before)
            record_dissipation(EN_WALL_DISS, before,
                               particle_energy(i, S, W, mass, Kin, 0.0), en, cnt)
            schedule(i, t, S, cellof, head, nxt, ev_t, ev_type, ev_p, tree, P,
                     IP, cnt, cp, ncp)
        else:   # EV_ADV
            advance(i, t, S, g)
            cnt[C_ADV] += 1
            schedule(i, t, S, cellof, head, nxt, ev_t, ev_type, ev_p, tree, P,
                     IP, cnt, cp, ncp)


@njit(cache=True)
def init_events(ev_t, ev_type, ev_p, tree, IP, t0):
    """All particles must have state time t0.  Every particle gets an
    advancement event at t0; the main loop then predicts them one by one."""
    N = IP[I_N]
    for i in range(N):
        ev_t[i] = t0
        ev_type[i] = EV_ADV
        ev_p[i] = -1
    ev_t[N] = INF
    cbt_build(tree, ev_t, N, IP[I_M])


@njit(cache=True)
def project(t, S, g, out_r, out_v):
    N = out_r.shape[0]
    for i in range(N):
        dt = t - S[i, ST]
        out_r[i, 0] = S[i, SX] + S[i, SVX] * dt
        out_r[i, 1] = S[i, SY] + S[i, SVY] * dt
        out_r[i, 2] = S[i, SZ] + (S[i, SVZ] - 0.5 * g * dt) * dt
        out_v[i, 0] = S[i, SVX]
        out_v[i, 1] = S[i, SVY]
        out_v[i, 2] = S[i, SVZ] - g * dt


@njit(cache=True)
def synchronize(t, S, g, N):
    for i in range(N):
        advance(i, t, S, g)


@njit(cache=True)
def rebase(T, S, ev_t, N, last):
    for i in range(N):
        S[i, ST] -= T
        last[i] -= T
        if ev_t[i] < INF:
            ev_t[i] -= T
    if ev_t[N] < INF:       # the kick slot
        ev_t[N] -= T


@njit(cache=True)
def restore_kick(kick_time, ev_t, ev_type, tree, IP):
    """Put the kick event back after init_events cleared it."""
    N = IP[I_N]
    ev_t[N] = kick_time
    ev_type[N] = EV_KICK
    cbt_update(N, tree, ev_t, IP[I_M])


@njit(cache=True)
def min_gap(r, rad, cellof, head, nxt, P, IP):
    """Minimum surface gap between neighbouring pairs for synchronized
    positions r.  Returns (min gap, i, j)."""
    nx = IP[I_NX]
    ny = IP[I_NY]
    nz = IP[I_NZ]
    N = IP[I_N]
    best = INF
    bi = -1
    bj = -1
    for i in range(N):
        c = cellof[i]
        ix = c % nx
        tmp = c // nx
        iy = tmp % ny
        iz = tmp // ny
        for jz in range(iz - 1, iz + 2):
            if jz < 0 or jz >= nz:
                continue
            for jy in range(iy - 1, iy + 2):
                if jy < 0 or jy >= ny:
                    continue
                for jx in range(ix - 1, ix + 2):
                    if jx < 0 or jx >= nx:
                        continue
                    j = head[jx + nx * (jy + ny * jz)]
                    while j >= 0:
                        if j > i:
                            dx = r[j, 0] - r[i, 0]
                            dy = r[j, 1] - r[i, 1]
                            dz = r[j, 2] - r[i, 2]
                            gap = np.sqrt(dx * dx + dy * dy + dz * dz) - rad[i] - rad[j]
                            if gap < best:
                                best = gap
                                bi = i
                                bj = j
                        j = nxt[j]
    return best, bi, bj


@njit(cache=True)
def check_cells(r, cellof, P, IP):
    """Number of particles whose centre is outside its assigned cell by more
    than 1e-9 (positions r synchronized)."""
    nx = IP[I_NX]
    ny = IP[I_NY]
    nz = IP[I_NZ]
    hx = P[P_HX]
    hz = P[P_HZ]
    z0 = P[P_Z0]
    bad = 0
    for i in range(r.shape[0]):
        c = cellof[i]
        ix = c % nx
        tmp = c // nx
        iy = tmp % ny
        iz = tmp // ny
        tol = 1e-9
        if r[i, 0] < ix * hx - tol or r[i, 0] > (ix + 1) * hx + tol:
            bad += 1
        elif r[i, 1] < iy * hx - tol or r[i, 1] > (iy + 1) * hx + tol:
            bad += 1
        elif r[i, 2] < z0 + iz * hz - tol:
            bad += 1
        elif iz < nz - 1 and r[i, 2] > z0 + (iz + 1) * hz + tol:
            bad += 1
    return bad


@njit(cache=True)
def check_events(S, ev_t, ev_type, ev_p, N, t_now):
    """Consistency of the event queue: returns number of violations."""
    bad = 0
    for i in range(N):
        if ev_t[i] < t_now:
            bad += 1
        if S[i, ST] > t_now:
            bad += 1
        if ev_type[i] == EV_PP:
            j = ev_p[i]
            if ev_type[j] != EV_PP or ev_p[j] != i or ev_t[j] != ev_t[i]:
                bad += 1
    return bad

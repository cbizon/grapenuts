# Column pilot, 2026-11-11

Numbers are from `results/column_pilot.json`, written by
`scripts/column_pilot.py` on branch `column-pilot`.

## Setup

- A vertical column of 10 spheres (`granular/column.py`), driven as a layer
  10 deep at Gamma = 5, f* = 0.44 (plate period 7.19).
- 200 cycles, three seeds (sphere diameters), 20 kicks per contact time
  unless stated.
- Triggers: `always` (every touch is a spring), `tc` (a touch is a spring if
  either sphere collided within t_c or holds a spring), and `tc` with springs
  made only at approach speeds below 0.1 or 0.01.

## Results

- **Hard spheres stall; every spring variant survives.** The hard-sphere
  column used up 20 million touches at cycle 1.81 at all three seeds. All 48
  runs with springs completed 200 cycles.
- **Overlap is set by impacts, not by resting load.** Largest overlap over the
  three seeds, in diameters:

  | stiffness k | t_c | always | tc | tc, springs below 0.1 | tc, springs below 0.01 |
  | --- | --- | --- | --- | --- | --- |
  | 6e4 | 9.1e-3 | 0.057 | 0.046 | 0.039 | 0.029 |
  | 6e5 | 2.9e-3 | 0.018 | 0.013 | 0.011 | 0.0078 |
  | 6e6 | 9.1e-4 | 0.0053 | 0.0039 | 0.0039 | 0.0046 |
  | 6e7 | 2.9e-4 | 0.0016 | 0.0013 | 0.0018 | 0.00097 |

  It falls as k^-1/2, the impact scaling, in every column. Making springs only
  at low approach speed lowers it by less than a factor of two: a hard hit on
  a sphere that already holds a spring compresses that spring at the hit's
  speed.
- **Cost: spring force evaluations per sphere per cycle** (mean of seeds):

  | stiffness k | always | tc | tc, springs below 0.1 | tc, springs below 0.01 |
  | --- | --- | --- | --- | --- |
  | 6e4 | 470 | 374 | 741 | 946 |
  | 6e5 | 1,470 | 1,330 | 2,760 | 4,390 |
  | 6e6 | 3,370 | 3,760 | 8,630 | 15,300 |
  | 6e7 | 10,800 | 10,100 | 27,900 | 51,000 |

  Hard collisions per sphere per cycle alongside: 4 to 52 for `tc`, 34 to 69
  with springs below 0.1, 100 to 139 with springs below 0.01.
- **Energy error is second order in the step.** For `tc` at k = 6e5 the
  unaccounted energy over 200 cycles, as a fraction of energy throughput, is
  -4e-3, -1e-3, -2e-4 and -6e-5 at 10, 20, 40 and 80 kicks per contact time
  (a loss). It does not depend on k at fixed kicks per contact time.
- **The speed-limited triggers gain energy.** At 20 kicks per contact time
  the residual is +3e-3 to +5e-3 with springs below 0.1 and +6e-4 to +1.6e-3
  with springs below 0.01. A separate check at k = 6e5 showed this also falls
  as the square of the step (3e-3 at 20 kicks per contact time, 3e-5 at 320).
- **The trigger changes the mean height.** Time-averaged centre-of-mass
  height, range over seeds: 14.5 to 15.1 for `always` and 14.1 to 15.0 for
  `tc` at k >= 6e5, against 13.5 to 13.7 for both speed-limited triggers.
  There is no hard-sphere baseline to say which is right, because that model
  stalls here.

## Caveats

- One dimension is the harshest case for impacts: approach speeds at spring
  creation reached 14 to 24, far above anything in a layer, and the column
  spends most of the cycle in flight (springs are active for a few percent
  of kick times). Neither the overlap nor the cost transfers to a 3D layer
  as a number; the scalings should.
- CPU times in the JSON are dominated by the per-kick loop over a 10-sphere
  column and say nothing about the cost in the event-driven engine.

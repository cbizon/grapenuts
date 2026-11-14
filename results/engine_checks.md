# Engine spring contacts: checks and 2D layer pilot, 2026-11-13

Branch `engine-springs`. Numbers are in `results/validation.json`
(`scripts/validation_data.py`) and `results/layer_pilot.json`
(`scripts/layer_pilot.py`). `scripts/build_report.py` turns both into the
charts page; its summary text is `results/report_summary.html`.

## Single contacts

- Contact time within 2% and peak overlap within 0.4% of the linear
  spring-dashpot law, k = 1e4 to 1e8, speeds 0.1 to 4.
- Restitution against the paper's e(v): within 0.6% at 50 kicks per contact
  time, up to 3.2% too high at 20 (0.722 against 0.700).
- Tangential, as outgoing against incoming surface speed over normal speed,
  mu = 0.5, kt/kn = 0.125: 0.125 -> -0.042 (Walton -0.044); 1 -> -0.194
  (-0.35); 2 -> -0.04 (-0.70); 3 -> 0.45 (0.025); 5 -> 2.10 (2.03);
  8 -> 5.04 (5.03). The spring matches Walton near head-on and in sliding,
  not between; no kt/kn matches throughout.
- A sphere set sliding on a static plate ends rolling at 5/7 of its speed.
- Layer energy residual (tc, k = 1e6, 100 spheres, 30 cycles): -1.7e-3 at 10
  kicks per contact time, -1.05e-3 at 20, -2.4e-4 at 80.

## 2D layer, L = 100, three seeds, 20 + 100 cycles

Gamma 3.5, f* 0.5, 5 layers (hard engine runs). Mean height, kinetic energy
per sphere, largest overlap, CPU seconds per run:

| method | height | kinetic | overlap | CPU s |
| --- | --- | --- | --- | --- |
| hard | 4.517 | 1.069 | none | 1.2 |
| tc k=1e5 | 4.557 | 1.107 | 0.017 | 6.5 |
| tc k=1e6 | 4.524 | 1.076 | 0.0052 | 4.7 |
| tc k=1e7 | 4.519 | 1.071 | 0.0016 | 2.8 |
| tc k=1e6, springs below 0.1 | 4.515 | 1.069 | 0.0025 | 1.4 |
| always k=1e6 | 4.522 | 1.079 | 0.0055 | 14.9 |

Seed standard deviation is about 0.004 in height and 0.002 in kinetic energy.

Gamma 5, f* 0.3, 10 layers (hard engine stalled before cycle 3 at 3 of 3
seeds, 300 million events in one twentieth of a cycle, about 335 CPU s each):

| method | height | kinetic | overlap | CPU s | energy residual |
| --- | --- | --- | --- | --- | --- |
| tc k=1e5 | 23.78 | 14.46 | 0.065 | 108 | -3.3e-3 |
| tc k=1e6 | 23.59 | 14.88 | 0.016 | 233 | -2.8e-3 |
| tc k=1e7 | 23.57 | 15.03 | 0.0048 | 459 | -2.2e-3 |
| tc k=1e6, springs below 0.1 | 23.57 | 14.98 | 0.015 | 199 | +2.1e-3 |
| always k=1e6 | 23.66 | 14.78 | 0.017 | 262 | -4.2e-3 |

All 15 runs with springs completed 120 cycles. Approach speeds at spring
creation reach 24 to 29 here. Kinetic energy is still rising with k.

## After the Walton collision phase (commit 7cda4dc), same day

Springs deliver Walton's tangential impulse during a collision and use the
tangential spring only when a contact outlasts an isolated one. Both result
files were regenerated with this law; the tables above are the earlier
spring-only law.

- Tangential, k = 1e8: largest difference from the hard operator 0.0012 in
  outgoing tangential over normal speed, over mu = 0.25, 0.5, 1 and ratios up
  to 8.
- At k = 1e6 and 20 kicks per contact time, mu = 0.5: largest difference 0.014
  at closing speed 0.1, 0.16 at 1, 0.71 at 4, each at the most grazing angle.
- Layer energy residual (tc, k = 1e6, 100 spheres, 30 cycles): -1.2e-3 at 10
  kicks per contact time, -3.2e-4 at 20, -3.7e-5 at 80.

Gamma 3.5, f* 0.5, 5 layers:

| method | height | kinetic | overlap | energy residual |
| --- | --- | --- | --- | --- |
| hard | 4.517 | 1.069 | none | 1e-14 |
| tc k=1e5 | 4.555 | 1.105 | 0.018 | -7.4e-4 |
| tc k=1e6 | 4.524 | 1.076 | 0.0053 | -2.6e-4 |
| tc k=1e7 | 4.518 | 1.070 | 0.0018 | -3.4e-5 |
| tc k=1e6, springs below 0.1 | 4.516 | 1.067 | 0.0036 | -5.0e-7 |
| always k=1e6 | 4.528 | 1.083 | 0.0056 | -1.4e-3 |

Gamma 5, f* 0.3, 10 layers (hard stalled at 3 of 3 seeds, as before):

| method | height | kinetic | overlap | CPU s | energy residual |
| --- | --- | --- | --- | --- | --- |
| tc k=1e5 | 23.77 | 14.82 | 0.068 | 112 | -1.9e-3 |
| tc k=1e6 | 23.68 | 14.89 | 0.016 | 225 | -1.5e-3 |
| tc k=1e7 | 23.50 | 15.09 | 0.0048 | 497 | -1.1e-3 |
| tc k=1e6, springs below 0.1 | 23.53 | 14.92 | 0.015 | 198 | +1.6e-3 |
| always k=1e6 | 23.64 | 15.01 | 0.017 | 266 | -2.1e-3 |

Seed standard deviations here are 0.02 to 0.08 in height and 0.07 to 0.15 in
kinetic energy. The dependence on k (about 1% in height, 2% in kinetic
energy over two decades) is not removed by the friction change.

## Time step or softness? (`scripts/steps_check.py`, `results/steps_check.json`)

The stalling condition again (Gamma 5, f* 0.3, 10 layers, same seeds and
protocol, Walton collision phase), at 50 kicks per contact time instead of 20.
Mean and standard deviation over three seeds:

| k | kicks per contact | height | kinetic | energy residual | CPU s |
| --- | --- | --- | --- | --- | --- |
| 1e5 | 20 | 23.77 ± 0.07 | 14.82 ± 0.09 | -1.9e-3 | 112 |
| 1e5 | 50 | 23.80 ± 0.06 | 14.44 ± 0.16 | -6.5e-4 | 267 |
| 1e6 | 20 | 23.68 ± 0.03 | 14.89 ± 0.15 | -1.5e-3 | 225 |
| 1e6 | 50 | 23.68 ± 0.08 | 14.99 ± 0.07 | -5.0e-4 | 533 |
| 1e7 | 20 | 23.50 ± 0.08 | 15.09 ± 0.11 | -1.1e-3 | 497 |

- A finer time step does not move either stiffness toward the k = 1e7 values.
  Height is unchanged at both. Kinetic energy at k = 1e5 moves away (14.82 to
  14.44); at k = 1e6 it changes within scatter.
- So the dependence on k is softness, not the time step. At k = 1e5 the
  20-kick restitution error (springs too bouncy) was offsetting part of it.
- The energy residual falls about threefold from 20 to 50 kicks.
- k = 1e7 was not rerun at 50 kicks.

## Where the dependence on stiffness levels off (`scripts/stiff_check.py`, `results/stiff_check.json`)

Stalling condition, same seeds and protocol, tc trigger, Walton collision
phase. Mean and standard deviation over three seeds; all runs completed 120
cycles.

| k | kicks per contact | height | kinetic | largest overlap | energy residual | CPU s |
| --- | --- | --- | --- | --- | --- | --- |
| 1e5 | 20 | 23.77 ± 0.07 | 14.82 ± 0.09 | 0.068 | -1.9e-3 | 112 |
| 1e6 | 20 | 23.68 ± 0.03 | 14.89 ± 0.15 | 0.016 | -1.5e-3 | 225 |
| 1e7 | 20 | 23.50 ± 0.08 | 15.09 ± 0.11 | 0.0048 | -1.1e-3 | 497 |
| 1e8 | 20 | 23.45 ± 0.05 | 14.96 ± 0.31 | 0.0014 | -6.4e-4 | 949 |
| 1e5 | 50 | 23.80 ± 0.06 | 14.44 ± 0.16 | 0.066 | -6.5e-4 | 267 |
| 1e6 | 50 | 23.68 ± 0.08 | 14.99 ± 0.07 | 0.014 | -5.0e-4 | 533 |
| 1e7 | 50 | 23.48 ± 0.05 | 14.96 ± 0.22 | 0.0045 | -3.6e-4 | 1209 |

- Height levels off between k = 1e7 and 1e8: 23.50 and 23.45, a difference
  smaller than the seed scatter. k = 1e6 is about 0.2 (0.9%) above that and
  k = 1e5 about 0.3 (1.3%).
- Kinetic energy is 14.9 to 15.1 from k = 1e6 up, with no trend the three
  seeds can resolve (one k = 1e8 seed gave 14.52). Only k = 1e5 at 50 kicks
  (14.44) stands clearly apart.
- k = 1e7 at 50 kicks agrees with k = 1e7 at 20 kicks within scatter.
- Hard collisions per sphere per cycle rise with k (28, 56, 129, 357) as fewer
  touches fall inside the contact-time window.

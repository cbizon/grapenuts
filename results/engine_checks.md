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

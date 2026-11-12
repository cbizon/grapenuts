# grapenuts

Hybrid hard-sphere / soft-sphere simulation of vertically oscillated granular
layers. An exploration, separate from `sequel`.

## The idea

A hard-sphere code can only express a contact as an instantaneous collision,
so a lasting contact becomes an endless stream of ever-smaller bounces. A
soft-sphere code expresses a contact as a spring force. Here each contact
chooses:

- a clean two-body collision stays an instantaneous hard collision;
- a contact where either sphere collided within the last contact duration
  `t_c`, or already holds a spring, becomes a spring for as long as it lasts.

The spring force is applied as small impulses ("kicks") every `dt`. Between
kicks every sphere is in free flight under gravity, so the event-driven
machinery (collision prediction, plate solver, cell grid) is unchanged.

The spring stiffness `k` is the one physical parameter: it sets
`t_c = pi sqrt(m_eff / k)`. As `k` grows the method tends to the hard-sphere
engine.

## Status

- `granular/column.py`: a one-dimensional prototype (a column of spheres on
  the plate) with the three triggers. Tests in `tests/test_column.py`.
- `results/column_pilot.md`: the first pilot, overlap, energy error and cost
  against stiffness, time step and trigger.
- The springs are not in the event-driven engine (`granular/edmd.py`) yet.

## Provenance

`granular/{__init__,edmd,sim,runner,export}.py` and the tests
`test_edmd`, `test_plate_root`, `test_reference`, `test_planar` and
`test_energy_audit` were copied unchanged from `sequel` at commit `6eae1cb`
(`~/Projects/experiments/sequel`, remote `cbizon/OscillationOverthruster`).
That engine is the event-driven hard-sphere code for Bizon et al., PRL 80, 57
(1998).

## Setup

```bash
uv sync
uv run pytest
```

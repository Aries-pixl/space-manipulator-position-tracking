# Validation of the small release

Checked on Windows with Python 3.12 and the versions in `requirements.txt`,
21 October 2026. The scientific environment also passed `pip check`.

- The four archived NPZ files were copied byte for byte from the original local
  research-data archive. Their hashes and original summary rows are included.
- `run_demo.py analyze` verified all supplied checksums, recomputed metrics,
  checked applied torque/increment bounds and solver-fallback input holding,
  and reproduced the original criterion classifications.
- `run_demo.py simulate` ran all four eight-second examples from the model and
  controller, without reading the archived trajectories. All completed. Their
  reported metrics matched the archived example metrics on this run.
- Model checks returned minimum mass-matrix eigenvalue 0.0229648910, symmetry
  error 3.05e-16, kinetic identity error 1.66e-12 and Jacobian finite-difference
  error 2.44e-11. These are numerical consistency checks at the tested state.
- The generated tracking plot was inspected for readable axes and method labels.

| Method | RMS (mm) | Final-second mean (mm) | Integral of squared torque (N² m² s) | Criterion |
|---|---:|---:|---:|---|
| MPC | 0.709706 | 0.724090 | 26.463400 | Fail |
| MPC + P | 1.760468 | 3.338601 | 5.486433 | Fail |
| MPC + R | 0.345388 | 0.298216 | 4.487997 | Pass |
| MPC + P + R | 0.336724 | 0.524473 | 1.788698 | Pass |

These are individual seed-100 results. The first two trajectories complete but
violate soft state constraints; their failing classifications are intentional.
This is not a guarantee of controller stability, real-time feasibility, identical
outputs on other machines, or absence of software defects. This release was not
used to rerun every experiment in the paper. The recorded solver times may vary,
and OSQP's timing-dependent adaptation can also alter fresh trajectories.

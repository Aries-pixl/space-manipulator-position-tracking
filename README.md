# Space manipulator position tracking

Core implementation and a small example-data subset for *Constrained Predictive
Position Tracking for Space Manipulator Precontact Approach*.

This package runs the six-joint model and constrained controller and checks one
fixed-gain ablation example. It is **not the complete numerical archive** for the
paper's tables, confidence intervals or supplementary studies.

## Install

Python 3.12 was used on Windows. From this repository directory:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

On Linux/macOS use `.venv/bin/python`; those platforms have not been tested here.
The first simulation compiles the Numba model and can take longer.

## Check the supplied records

```powershell
.venv\Scripts\python run_demo.py analyze
```

This verifies the data hashes, recomputes tracking error and control effort, checks
input constraints and endpoint classifications, and writes `metrics.csv` plus
`tracking.png` under `runs/archived`. It does not simulate new trajectories.

## Run the example again

```powershell
.venv\Scripts\python run_demo.py simulate
```

This runs MPC, MPC+P, MPC+R and MPC+P+R sequentially for eight seconds each, then
saves fresh trajectories, metadata, metrics and a comparison plot under
`runs/fresh`. An output directory must be new or empty. To repeat a run, supply
another directory, for example `--output runs/fresh_2`. No archived trajectories
are used as simulation results.

`example_config.json` records the common seed (100), gain (4), acceleration penalty
(0.002), inertia uncertainty (0.2), disturbance coefficient (0.12) and duration.
The DH convention, masses, inertia tensors, centres of mass, target construction,
plant integration, QP constraints and estimator parameters are defined in
`experiment.py`. This file is copied without numerical changes from the validated
implementation. `run_demo.py` is only the reduced execution and analysis entry point.

## Included data and scope

The four NPZ files are the original `uncertain` records at the first evaluation
seed, 100, one per fixed-gain MPC variant. They were selected by seed and method,
not by success or error magnitude. Both passing and failing endpoint
classifications are retained. `reference_metrics.json` contains their original
summary rows; `SHA256.json` verifies this subset. These are individual examples,
not 20-seed averages and not the independently tuned baseline comparison.

Load NPZ files with `numpy.load(..., allow_pickle=False)`. Each contains 800
samples at 0.01 s intervals: `time` (s), `q` (rad), `v` (rad/s), `tau` (N m),
`delta` (N m per update), and `position`/`reference` (m). Joint arrays have six
columns; Cartesian arrays have three. Solver diagnostics use the `qp_` prefix.
The `position_criterion` also requires no recorded joint-angle or speed violation;
a small position error alone does not imply a passing classification.

The public subset omits the other seeds, tuning sweeps, sensing/base/deadline
studies and joint-limit investigations. The manuscript's full statistics cannot
be independently reconstructed from these four records alone. Full records are
retained by the authors; this repository does not claim they are publicly deposited.

The controlled output is end-effector position. Soft state constraints do not
guarantee hard state safety, and the scalar Gaussian features have limited
expressiveness. The study does not demonstrate orientation capture, physical
contact or reliable 100 Hz deployment. Solver adaptation depends partly on
measured timing, so fresh trajectories and timings need not match bit for bit.
See `VALIDATION.md` for checks performed on this small package.

## Citation and rights

Version: 1.0.0 (tag `v1.0.0`).

Source repository: https://github.com/Aries-pixl/space-manipulator-position-tracking

Versioned example-code/data archive: 10.5281/zenodo.23526932

 The archive DOI is not a journal article DOI. Author metadata are supplied in `CITATION.cff`.

No additional software/data reuse license has been selected for this prepared
package. Public access alone does not grant an MIT or Creative Commons license.
Third-party libraries retain their own licenses and are installed rather than redistributed.

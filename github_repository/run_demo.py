"""Analyze four archived examples or rerun the same fixed-gain comparison."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os

# Keep numerical jobs sequential and avoid nested BLAS parallelism.
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
LABELS = {'mpc': 'MPC', 'mpc_posture': 'MPC + P',
          'mpc_residual': 'MPC + R', 'full': 'MPC + P + R'}

def summarize(path, reference):
    with np.load(path, allow_pickle=False) as d:
        t, q, v, u = d['time'], d['q'], d['v'], d['tau']
        for key in ('time', 'q', 'v', 'tau', 'position', 'reference'):
            if not np.all(np.isfinite(d[key])):
                raise ValueError(f'Nonfinite recorded {key}: {path.name}')
        np.testing.assert_allclose(t, np.arange(len(t)) * 0.01, atol=1e-12)
        delta = u - np.vstack([np.zeros((1, 6)), u[:-1]])
        np.testing.assert_allclose(delta, d['delta'], atol=1e-12)
        if np.max(abs(u)) > 8 + 1e-8 or np.max(abs(delta)) > 1 + 1e-8:
            raise ValueError(f'Applied-input bound violated: {path.name}')
        held = d['qp_fallback'].astype(bool)
        if np.max(abs(delta[held]), initial=0) > 1e-12:
            raise ValueError(f'Fallback did not hold applied input: {path.name}')
        error = np.linalg.norm(d['position'] - d['reference'], axis=1)
        bad = np.any((abs(q) > 2.6 + 1e-8) | (abs(v) > 2 + 1e-8), axis=1)
        # Legacy archived rows omit completion; the full 800 samples establish it.
        complete = bool(reference.get('completed', len(t) == 800)) and len(t) == 800
        passed = bool(complete and error[-100:].mean() < .01
                      and error[-100:].max() < .02 and not np.any(bad))
        row = dict(method=reference['method'], seed=reference['seed'], completed=complete,
                   rms_mm=float(1000 * np.sqrt(np.mean(error ** 2))),
                   terminal_mm=float(1000 * np.mean(error[-100:])),
                   effort_Nm2s=float(.01 * np.sum(u * u)),
                   state_violation_pct=float(100 * np.mean(bad)),
                   joint_margin_rad=float(2.6 - np.max(abs(q))),
                   position_criterion=passed)
        for key in ('rms_mm', 'terminal_mm', 'effort_Nm2s',
                    'state_violation_pct', 'joint_margin_rad'):
            if not np.isclose(row[key], reference[key], rtol=1e-10, atol=1e-8):
                raise ValueError(f'Metric mismatch: {path.name}, {key}')
        if passed != reference['position_criterion']:
            raise ValueError(f'Endpoint classification mismatch: {path.name}')
        return row, (t.copy(), 1000 * error)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['analyze', 'simulate'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    out = (args.output or ROOT / 'runs' / ('archived' if args.mode == 'analyze' else 'fresh')).resolve()
    if ROOT.is_relative_to(out) or out.is_relative_to(ROOT / 'data'):
        parser.error('Output must not contain or replace the source or archived data.')
    if out.exists() and any(out.iterdir()):
        parser.error('Output directory must be new or empty.')
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / 'example_config.json').read_text(encoding='utf-8'))
    if args.mode == 'analyze':
        data = ROOT / 'data'
        hashes = json.loads((data / 'SHA256.json').read_text(encoding='utf-8'))
        for name, expected in hashes.items():
            if hashlib.sha256((data / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Data checksum mismatch: {name}')
        references = json.loads((data / 'reference_metrics.json').read_text(encoding='utf-8'))
        sources = [(data / f"uncertain_{r['method']}_100.npz", r) for r in references]
    else:
        from experiment import run, validate
        checks = validate()
        (out / 'model_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
        sources = []
        for method in config['methods']:
            print(f'Running {LABELS[method]}, seed {config["seed"]} ...', flush=True)
            path = out / f'uncertain_{method}_{config["seed"]}.npz'
            row = run(config['case'], method=method, seed=config['seed'],
                      gain=config['gain'], effort=config['effort'], save=path)
            path.with_suffix('.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
            sources.append((path, row))
    rows = []
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for path, ref in sources:
        row, (t, e) = summarize(path, ref)
        rows.append(row)
        suffix = '' if row['completed'] else ' (terminated prefix)'
        ax.plot(t, e, label=LABELS[row['method']] + suffix, linewidth=1.2)
    with (out / 'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ax.set(xlabel='Time (s)', ylabel='Position error (mm)',
           title='Fixed-gain example, seed 100')
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(out / 'tracking.png', dpi=200)
    plt.close(fig)
    (out / 'run_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(json.dumps(rows, indent=2))
    print(f'Outputs: {out}')

if __name__ == '__main__':
    main()

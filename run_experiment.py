"""Full trainable-output-weights experiment (runs A-D), extending the pilot.

Usage (from this directory):
    python run_experiment.py test          # self-tests only
    python run_experiment.py A             # main run: 10,000 He + 32 box, eta=0.05, 1e7 steps
    python run_experiment.py B             # rate run: 200 He pairs from A + 32 box, 1e9 steps
    python run_experiment.py C             # output-weight initialisation variants
    python run_experiment.py D             # learning-rate sensitivity
    python analyze.py                      # figures + summary from results_*/

Model, data, loss, optimiser and margin are identical to the pilot
(../Trainable_Output_Weights_Pilot): f(x)=v1 ReLU(w1 x+b1)+v2 ReLU(w2 x+b2),
data (-1,-1),(1,1), mean exponential loss, simultaneous full-batch GD in float64.
Runs whose loss is not below 0.5 at step 10,000 are stopped (status 1).
Parallelism: one compiled kernel call per run, dispatched from a thread pool
(ctypes releases the GIL), so OpenMP is not required.
"""
import argparse
import csv
import ctypes
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PAPER_RUNS_ZIP = ROOT.parent/'gd 0.05.zip'   # paper's GD eta=0.05 run (1,000 He inits, seed 42)
FILTER_STEP, FILTER_LOSS = 10_000, 0.5
DECADES = [10, 100, 1000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000]
PARAM_NAMES = ['w1', 'b1', 'w2', 'b2', 'v1', 'v2']


# ----------------------------------------------------------------------------- model
def outputs(p, x):
    x = np.asarray(x)
    return p[4]*np.maximum(p[0]*x+p[1], 0) + p[5]*np.maximum(p[2]*x+p[3], 0)


def loss_grad(p):
    x = np.array([-1., 1.])
    w, b, v = p[[0, 2]], p[[1, 3]], p[4:6]
    z = x[:, None]*w+b
    r = np.maximum(z, 0.)
    f = r@v
    factors = np.exp(-x*f)
    d = -0.5*x*factors
    gh = d[:, None]*v*(z > 0)
    grad = np.empty(6)
    grad[[0, 2]] = (gh*x[:, None]).sum(0)
    grad[[1, 3]] = gh.sum(0)
    grad[4:6] = d@r
    return float(factors.mean()), grad


def exact_margin(p):
    """Distance from the nearest training point to the zero set (pilot routine, unchanged)."""
    if not np.all(np.isfinite(p)):
        return np.nan
    f = outputs(p, [-1., 1.])
    if not (f[0] < 0 < f[1]):
        return 0.
    cuts = sorted(set(-p[j+1]/p[j] for j in (0, 2) if p[j] != 0))
    edges = [-np.inf, *cuts, np.inf]
    distances = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mid = (lo+hi)/2 if np.isfinite(lo) and np.isfinite(hi) else (
            hi-max(1., abs(hi)) if np.isfinite(hi) else (
                lo+max(1., abs(lo)) if np.isfinite(lo) else 0.))
        active = np.array([p[0]*mid+p[1] > 0, p[2]*mid+p[3] > 0])
        slope = np.sum(p[4:6]*p[[0, 2]]*active)
        intercept = np.sum(p[4:6]*p[[1, 3]]*active)
        if slope == 0.:
            if intercept == 0.:
                distances.extend(abs(x-np.clip(x, lo, hi)) for x in (-1., 1.))
        else:
            root = -intercept/slope
            tol = 1e-12*(1+abs(root))
            if lo-tol <= root <= hi+tol:
                root = np.clip(root, lo, hi)
                distances.extend(abs(x-root) for x in (-1., 1.))
    if not distances:
        raise AssertionError('Opposite labels but no zero found')
    margin = min(distances)
    if not -1e-10 <= margin <= 1+1e-10:
        raise AssertionError(('Invalid margin', p, margin))
    return float(np.clip(margin, 0., 1.))


def _measure_block(block):
    """Loss and exact margin for an array (runs, checkpoints, 6). Worker for a process pool."""
    losses = np.full(block.shape[:2], np.nan)
    margins = losses.copy()
    for i in range(block.shape[0]):
        for j in range(block.shape[1]):
            p = block[i, j]
            if np.all(np.isfinite(p)):
                losses[i, j] = loss_grad(p)[0]
                margins[i, j] = exact_margin(p)
    return losses, margins


# ----------------------------------------------------------------------------- kernel
def build_kernel():
    build = ROOT/'build'
    build.mkdir(exist_ok=True)
    lib = build/'train_kernel.so'
    cxx = os.environ.get('CXX', 'c++')
    cmd = [cxx, '-O3', '-std=c++17', '-shared', '-fPIC', str(ROOT/'train_kernel.cpp'), '-o', str(lib)]
    subprocess.run(cmd, check=True)
    kernel = ctypes.CDLL(str(lib)).train_batch
    dptr = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    iptr = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    lptr = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
    kernel.argtypes = [dptr, dptr, iptr, ctypes.c_int, lptr, ctypes.c_int,
                       ctypes.c_int64, ctypes.c_double, ctypes.c_int, dptr, iptr, lptr]
    kernel.restype = None
    return kernel, cmd


def run_one(kernel, p, lr, mode, checkpoints, filter_step=FILTER_STEP):
    k = len(checkpoints)
    snaps = np.full((1, k, 6), np.nan)
    status = np.zeros(1, dtype=np.int32)
    stop = np.zeros(1, dtype=np.int64)
    kernel(np.ascontiguousarray([p], dtype=np.float64), np.array([lr], dtype=np.float64),
           np.array([mode], dtype=np.int32), 1, checkpoints, k, filter_step, FILTER_LOSS, 1,
           snaps, status, stop)
    return snaps[0], int(status[0]), int(stop[0])


def run_all(kernel, initial, lrs, modes, checkpoints, workers, label):
    n, k = len(initial), len(checkpoints)
    checkpoints = np.ascontiguousarray(checkpoints, dtype=np.int64)
    params = np.full((n, k, 6), np.nan)
    status = np.zeros(n, dtype=np.int32)
    stops = np.zeros(n, dtype=np.int64)
    t0 = time.perf_counter()
    done = 0
    with ThreadPoolExecutor(workers) as ex:
        futures = {ex.submit(run_one, kernel, initial[i], lrs[i], modes[i], checkpoints): i
                   for i in range(n)}
        for fut in as_completed(futures):
            i = futures[fut]
            params[i], status[i], stops[i] = fut.result()
            done += 1
            if done % max(1, n//20) == 0 or done == n:
                print(f'[{label}] {done}/{n} runs, {time.perf_counter()-t0:.0f}s', flush=True)
    return params, status, stops, time.perf_counter()-t0


# ----------------------------------------------------------------------------- tests
def self_test(kernel):
    rng = np.random.default_rng(8642)
    max_fd = max_ref = 0.
    for trial in range(30):
        p = rng.normal(size=6)
        _, grad = loss_grad(p)
        eps = 1e-6
        fd = np.empty(6)
        for j in range(6):
            s = np.eye(6)[j]*eps
            fd[j] = (loss_grad(p+s)[0]-loss_grad(p-s)[0])/(2*eps)
        err = float(np.max(abs(fd-grad)/(1+abs(grad))))
        max_fd = max(max_fd, err)
        assert err < 1e-6, (trial, err)
        p = rng.normal(scale=0.3, size=6)
        for mode in (0, 1):
            out, st, _ = run_one(kernel, p, .01, mode, np.array([0, 100], dtype=np.int64), filter_step=-1)
            ref = p.copy()
            for _ in range(100):
                _, g = loss_grad(ref)
                if not mode:
                    g[4:] = 0
                ref -= .01*g
            err = float(np.max(abs(ref-out[-1])))
            max_ref = max(max_ref, err)
            assert st == 0 and err < 1e-12, err
    assert abs(exact_margin(np.array([1., -.2, -1., .2, 1., -1.]))-.8) < 1e-12
    assert abs(exact_margin(np.array([1., -.2, -1., -.3, 1., -1.]))-.7) < 1e-12
    assert exact_margin(np.array([1., 0., -1., 0., 1., -1.])) == 1.
    from scipy.optimize import brentq
    checked = 0
    for _ in range(500):
        p = rng.normal(size=6)
        f = outputs(p, [-1., 1.])
        if f[0] < 0 < f[1]:
            xs = np.linspace(-2, 2, 4001)
            fs = outputs(p, xs)
            roots = list(xs[fs == 0])
            for i in np.flatnonzero(fs[:-1]*fs[1:] < 0):
                roots.append(brentq(lambda x: outputs(p, x), xs[i], xs[i+1]))
            ref = min(min(abs(-1-r), abs(1-r)) for r in roots)
            assert abs(exact_margin(p)-ref) < .0011, (p, exact_margin(p), ref)
            checked += 1
    tests = dict(finite_difference_max_relative_error=max_fd,
                 compiled_vs_numpy_max_absolute_error=max_ref,
                 independently_checked_random_margins=checked)
    tests.update(paper_init_check())
    return tests


def paper_init_check():
    """He initialisations must equal the paper's GD eta=0.05 run (w1_0, w2_0 columns)."""
    if not PAPER_RUNS_ZIP.exists():
        return dict(paper_init_check='skipped: archive not found')
    import io
    import zipfile
    with zipfile.ZipFile(PAPER_RUNS_ZIP) as z:
        rows = list(csv.DictReader(io.TextIOWrapper(z.open('experiment_5f_runs.csv'))))
    he = he_inits(len(rows))
    diff = max(max(abs(he[i, 0]-float(r['w1_0'])), abs(he[i, 2]-float(r['w2_0']))) for i, r in enumerate(rows))
    assert diff == 0.0, diff
    hits = sum(r['stop_reason'] == 'hit condition' for r in rows)
    return dict(paper_init_check_runs=len(rows), paper_init_max_abs_diff=diff, paper_run_hits=hits)


# ----------------------------------------------------------------------------- inits
def he_inits(n):
    """Same draws as the paper's main experiment: default_rng(42), w1 then w2 per run."""
    rng = np.random.default_rng(42)
    w = rng.normal(0., np.sqrt(2), (n, 2))
    p = np.zeros((n, 6))
    p[:, 0], p[:, 2] = w[:, 0], w[:, 1]
    p[:, 4:] = [1., -1.]
    return p


def box_inits(n=32):
    """Theorem-box draws; identical to the pilot's controlled cohort."""
    rng = np.random.default_rng(42)
    p = np.zeros((n, 6))
    p[:, 0] = rng.uniform(2.7, 3, n)
    p[:, 2] = -rng.uniform(0, .5, n)
    p[:, 4:] = [1., -1.]
    return p


def apply_vinit(p, vinit, cohort):
    p = p.copy()
    if vinit == 'unit':
        p[:, 4:] = [1., -1.]
    elif vinit == 'balanced':          # |v_j| = ||(w_j, b_j)||, signs (+, -)
        p[:, 4] = np.hypot(p[:, 0], p[:, 1])
        p[:, 5] = -np.hypot(p[:, 2], p[:, 3])
    elif vinit == 'random':            # v ~ N(0,1), arbitrary signs
        p[:, 4:] = np.random.default_rng([7, cohort == 'he_random']).normal(0., 1., (len(p), 2))
    elif vinit == 'small':
        p[:, 4:] = [.1, -.1]
    else:
        raise ValueError(vinit)
    return p


def make_checkpoints(steps):
    n = 161 if steps <= 10**7 else 201
    return np.unique(np.r_[0, np.geomspace(1, steps, n).astype(np.int64),
                           [d for d in DECADES if d <= steps]]).astype(np.int64)


def experiment_configs(exp):
    """Return (steps, list of (cohort, seed, lr, train_output, vinit, initial params))."""
    he_full, box = he_inits(10_000), box_inits(32)
    items = []
    if exp == 'A':
        steps = 10**7
        for cohort, starts in (('he_random', he_full), ('theorem_box', box)):
            for s, p in enumerate(starts):
                for mode in (0, 1):
                    items.append((cohort, s, .05, mode, 'unit', p))
    elif exp == 'B':
        steps = 10**9
        a = np.load(ROOT/'results_A'/'trajectories.npz')
        ok = {}
        for c, s, m, st in zip(a['cohort'], a['seed'], a['train_output'], a['status']):
            if c == 'he_random' and st == 0:
                ok.setdefault(int(s), set()).add(int(m))
        paired = sorted(s for s, m in ok.items() if m == {0, 1})
        chosen = sorted(np.random.default_rng(2026).choice(paired, size=min(200, len(paired)), replace=False))
        for cohort, seeds, starts in (('he_random', chosen, he_full), ('theorem_box', range(32), box)):
            for s in seeds:
                for mode in (0, 1):
                    items.append((cohort, int(s), .05, mode, 'unit', starts[s]))
    elif exp == 'C':
        steps = 10**7
        for vinit in ('unit', 'balanced', 'random', 'small'):
            for cohort, starts in (('he_random', he_full[:2000]), ('theorem_box', box)):
                for s, p in enumerate(apply_vinit(starts, vinit, cohort)):
                    items.append((cohort, s, .05, 1, vinit, p))
    elif exp == 'D':
        steps = 10**7
        for lr in (.01, .1):
            for cohort, starts in (('he_random', he_full[:2000]), ('theorem_box', box)):
                for s, p in enumerate(starts):
                    for mode in (0, 1):
                        items.append((cohort, s, lr, mode, 'unit', p))
    else:
        raise ValueError(exp)
    return steps, items


# ----------------------------------------------------------------------------- main
def run_experiment(exp, workers, kernel, compile_cmd, tests):
    steps, items = experiment_configs(exp)
    out = ROOT/f'results_{exp}'
    out.mkdir(exist_ok=True)
    checkpoints = make_checkpoints(steps)
    initial = np.array([it[5] for it in items])
    lrs = np.array([it[2] for it in items])
    modes = np.array([it[3] for it in items])
    print(f'[{exp}] {len(items)} runs, {steps:.0e} steps, {len(checkpoints)} checkpoints', flush=True)
    params, status, stops, train_s = run_all(kernel, initial, lrs, modes, checkpoints, workers, exp)

    # Loss and exact margin at every checkpoint of completed runs (process pool).
    t0 = time.perf_counter()
    losses = np.full(params.shape[:2], np.nan)
    margins = losses.copy()
    done_ids = np.flatnonzero(status == 0)
    chunks = np.array_split(done_ids, max(1, min(len(done_ids), 8*workers)))
    with ProcessPoolExecutor(workers) as ex:
        for ids, (l, m) in zip(chunks, ex.map(_measure_block, [params[c] for c in chunks])):
            losses[ids], margins[ids] = l, m
    measure_s = time.perf_counter()-t0
    gaps = 1-margins

    cfg = dict(cohort=np.array([it[0] for it in items]), seed=np.array([it[1] for it in items]),
               lr=lrs, train_output=modes, vinit=np.array([it[4] for it in items]))
    np.savez_compressed(out/'trajectories.npz', checkpoints=checkpoints, params=params, losses=losses,
                        margins=margins, gaps=gaps, status=status, stop_steps=stops,
                        initial=initial, **cfg)

    decs = [d for d in DECADES if d <= steps and d >= 1000]
    idx = {d: int(np.searchsorted(checkpoints, d)) for d in decs}
    with (out/'runs.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'cohort', 'seed', 'lr', 'train_output', 'vinit', 'status', 'stop_step',
                    *[n+'_0' for n in PARAM_NAMES], *[n+'_T' for n in PARAM_NAMES],
                    'loss_T', 'margin_T', *[f'gap_{d:.0e}' for d in decs]])
        for i, it in enumerate(items):
            last = params[i, -1] if status[i] == 0 else np.full(6, np.nan)
            w.writerow([i, it[0], it[1], it[2], it[3], it[4], int(status[i]), int(stops[i]),
                        *initial[i], *last, losses[i, -1], margins[i, -1],
                        *[gaps[i, idx[d]] for d in decs]])
    with (out/'checkpoints_decades.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'cohort', 'seed', 'lr', 'train_output', 'vinit', 'step', 'loss', 'margin', 'gap',
                    *PARAM_NAMES])
        cidx = [j for j, t in enumerate(checkpoints) if t == 0 or t in DECADES]
        for i in done_ids:
            it = items[i]
            for j in cidx:
                w.writerow([i, it[0], it[1], it[2], it[3], it[4], int(checkpoints[j]), losses[i, j],
                            margins[i, j], gaps[i, j], *params[i, j]])
    meta = dict(experiment=exp, steps=steps, runs=len(items), completed=int((status == 0).sum()),
                filter_failed=int((status == 1).sum()), nonfinite=int((status == 2).sum()),
                train_seconds=train_s, measure_seconds=measure_s, workers=workers,
                filter=f'loss < {FILTER_LOSS} at step {FILTER_STEP}; failures stopped',
                compile_command=compile_cmd, python=platform.python_version(), numpy=np.__version__,
                machine=platform.platform(), processor=platform.processor(), tests=tests,
                status_codes={'0': 'completed', '1': 'loss filter failed', '2': 'nonfinite'})
    (out/'meta.json').write_text(json.dumps(meta, indent=2, default=lambda x: x.item()))
    print(f'[{exp}] done: train {train_s:.0f}s, measure {measure_s:.0f}s, '
          f'completed {meta["completed"]}/{len(items)}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('exp', choices=['test', 'A', 'B', 'C', 'D', 'calibrate'])
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    args = ap.parse_args()
    kernel, cmd = build_kernel()
    tests = self_test(kernel)
    print('VALIDATION', json.dumps(tests), flush=True)
    if args.exp == 'calibrate':
        p = he_inits(10)[0]
        if p[0] < 0 or p[2] > 0:
            p = box_inits(1)[0]
        n = 10**8
        t0 = time.perf_counter()
        run_one(kernel, p, .05, 1, np.array([0, n], dtype=np.int64), filter_step=-1)
        dt = time.perf_counter()-t0
        print(f'single thread: {n/dt:.3e} steps/s', flush=True)
        return
    if args.exp != 'test':
        run_experiment(args.exp, args.workers, kernel, cmd, tests)


if __name__ == '__main__':
    main()

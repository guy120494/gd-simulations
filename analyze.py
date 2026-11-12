"""Figures and summary statistics for runs A-D (reads results_*/trajectories.npz).

Run: python analyze.py   ->  figures/*.png|pdf, summary.json, RESULTS.md
Comparisons between fixed and trainable output weights always use matched pairs:
the same initialisation, both runs completed (loss < 0.5 at step 10,000).
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
FIG = ROOT/'figures'
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
INK, MUTED = '#2b2b2a', '#6b6a64'
MODE_NAME = {0: 'Fixed output weights', 1: 'Trainable output weights'}
MODE_COLOR = {0: BLUE, 1: ORANGE}
COHORT_NAME = {'he_random': 'He initialisation', 'theorem_box': 'Theorem-box initialisation'}
RNG = np.random.default_rng(0)


def load(exp):
    f = ROOT/f'results_{exp}'/'trajectories.npz'
    return dict(np.load(f)) if f.exists() else None


def select(d, cohort, mode=None, lr=.05, vinit='unit', completed=True):
    m = (d['cohort'] == cohort) & np.isclose(d['lr'], lr) & (d['vinit'] == vinit)
    if mode is not None:
        m &= d['train_output'] == mode
    if completed:
        m &= d['status'] == 0
    return np.flatnonzero(m)


def pairs(d, cohort, lr=.05, seeds=None):
    """Indices (fixed, trainable) of matched completed pairs, ordered by seed."""
    a = {int(d['seed'][i]): i for i in select(d, cohort, 0, lr)}
    b = {int(d['seed'][i]): i for i in select(d, cohort, 1, lr)}
    common = sorted(a.keys() & b.keys())
    if seeds is not None:
        common = [s for s in common if s in seeds]
    return np.array([a[s] for s in common], int), np.array([b[s] for s in common], int)


def col(d, step):
    return int(np.searchsorted(d['checkpoints'], step))


def stats(x):
    x = np.asarray(x, float)
    return dict(n=int(len(x)), median=float(np.median(x)), q25=float(np.quantile(x, .25)),
                q75=float(np.quantile(x, .75)), mean=float(np.mean(x)), min=float(x.min()), max=float(x.max()))


def boot_ci(x, f=np.median, n=4000):
    x = np.asarray(x, float)
    b = [f(RNG.choice(x, len(x))) for _ in range(n)]
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def wilcoxon(x, y):
    from scipy.stats import wilcoxon as w
    return float(w(x, y).pvalue)


# ----------------------------------------------------------------------------- plotting helpers
def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e5e0',
                         'grid.linewidth': .6, 'legend.frameon': False, 'lines.linewidth': 2,
                         'savefig.dpi': 200, 'figure.facecolor': 'white'})
    return plt


def band(ax, t, q, color, label, ls='-'):
    ax.plot(t, np.median(q, 0), color=color, label=label, ls=ls)
    ax.fill_between(t, np.quantile(q, .25, 0), np.quantile(q, .75, 0), color=color, alpha=.15, lw=0)


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG/f'{name}.png', bbox_inches='tight')
    fig.savefig(FIG/f'{name}.pdf', bbox_inches='tight')


# ----------------------------------------------------------------------------- A
def analyze_A(d, plt):
    out = {}
    steps = [10**k for k in range(3, 8)]
    for cohort in ('he_random', 'theorem_box'):
        att = {m: len(select(d, cohort, m, completed=False)) for m in (0, 1)}
        ok = {m: set(d['seed'][select(d, cohort, m)].tolist()) for m in (0, 1)}
        f, t = pairs(d, cohort)
        row = dict(attempted=att[0], completed_fixed=len(ok[0]), completed_trainable=len(ok[1]),
                   completed_both=len(ok[0] & ok[1]), fixed_only=len(ok[0]-ok[1]),
                   trainable_only=len(ok[1]-ok[0]), checkpoints={})
        for s in steps:
            j = col(d, s)
            row['checkpoints'][str(s)] = dict(
                fixed_gap=stats(d['gaps'][f, j]), trainable_gap=stats(d['gaps'][t, j]),
                fixed_loss_median=float(np.median(d['losses'][f, j])),
                trainable_loss_median=float(np.median(d['losses'][t, j])))
        j5, j7 = col(d, 10**5), col(d, 10**7)
        gf, gt = d['gaps'][f, j7], d['gaps'][t, j7]
        row['terminal_paired'] = dict(
            frac_trainable_gap_larger=float(np.mean(gt > gf)),
            median_ratio_trainable_over_fixed=float(np.median(gt/gf)),
            median_ratio_ci95=boot_ci(gt/gf), wilcoxon_p=wilcoxon(gt, gf))
        for name, ids in (('fixed', f), ('trainable', t)):
            red = 1-d['gaps'][ids, j7]/d['gaps'][ids, j5]
            row[f'{name}_late_reduction_1e5_to_1e7'] = dict(
                median_relative_reduction=float(np.median(red)),
                frac_improved=float(np.mean(red > 0)))
            m5, m7 = np.median(d['gaps'][ids, j5]), np.median(d['gaps'][ids, j7])
            row[f'{name}_late_reduction_1e5_to_1e7']['reduction_of_median_gap'] = float(1-m7/m5)
        # output-weight behaviour in trainable runs
        p0, pT = d['params'][t, 0], d['params'][t, -1]
        v = d['params'][t][:, :, 4:]
        flips = np.any(np.sign(v) != np.sign(p0[:, None, 4:]), axis=(1, 2))
        bal0 = p0[:, 4:]**2-p0[:, [0, 2]]**2-p0[:, [1, 3]]**2
        balT = pT[:, 4:]**2-pT[:, [0, 2]]**2-pT[:, [1, 3]]**2
        row['trainable_output_weights'] = dict(
            v1_final=stats(pT[:, 4]), v2_final=stats(pT[:, 5]),
            runs_with_v_sign_change=int(flips.sum()),
            balance_neuron1_initial=stats(bal0[:, 0]), balance_neuron1_final=stats(balT[:, 0]),
            balance_neuron2_initial=stats(bal0[:, 1]), balance_neuron2_final=stats(balT[:, 1]),
            param_norm_final=stats(np.linalg.norm(pT, axis=1)),
            fixed_param_norm_final=stats(np.linalg.norm(d['params'][f, -1], axis=1)))
        out[cohort] = row
    # Consistency with the paper's run: fixed-mode successes among the first 1,000 He inits.
    first = select(d, 'he_random', 0)
    out['he_random']['fixed_success_first_1000'] = int(np.sum(d['seed'][first] < 1000))

    t = d['checkpoints']
    mask = t >= 10
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, cohort in zip(axes, ('he_random', 'theorem_box')):
        f, tr = pairs(d, cohort)
        for m, ids in ((0, f), (1, tr)):
            band(ax, t[mask], d['gaps'][ids][:, mask], MODE_COLOR[m], MODE_NAME[m])
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_title(f'{COHORT_NAME[cohort]} ({len(f):,} matched pairs)', fontsize=10, color=INK)
        ax.set_xlabel('GD updates')
    axes[0].set_ylabel('Margin gap  1 − margin')
    axes[0].legend(loc='lower left', fontsize=9)
    fig.text(.5, -.04, 'Median and interquartile band, η = 0.05. Pairs share the initialisation; '
             'both runs reach loss < 0.5 by step 10,000.', ha='center', fontsize=8.5, color=MUTED)
    fig.tight_layout()
    save(fig, 'A_gap_fixed_vs_trainable')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, cohort in zip(axes, ('he_random', 'theorem_box')):
        f, tr = pairs(d, cohort)
        x, y = d['gaps'][f, -1], d['gaps'][tr, -1]
        ax.scatter(x, y, s=10, color=ORANGE, alpha=.45, lw=0)
        lo, hi = min(x.min(), y.min())*.8, max(x.max(), y.max())*1.25
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls='--')
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.text(hi/2.2, hi/2.2/1.9, 'equal gap', fontsize=8, color=MUTED, ha='center', va='center',
                rotation=45)
        ax.set_xlabel('Gap, fixed output weights')
        ax.set_title(f'{COHORT_NAME[cohort].split()[0]}: trainable gap larger in {np.mean(y > x):.0%} of {len(x):,} pairs',
                     fontsize=10, color=INK)
    axes[0].set_ylabel('Gap, trainable output weights')
    fig.text(.5, -.02, 'Each point is one initialisation after 10⁷ GD updates (η = 0.05).',
             ha='center', fontsize=8.5, color=MUTED)
    fig.tight_layout()
    save(fig, 'A_paired_terminal_gap')
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------- B
def analyze_B(d, plt):
    out = {}
    t = d['checkpoints']
    win = (t >= 10**5) & (t <= 10**9)
    x = np.log(np.log(t[win]))
    decs = [10**k for k in range(4, 10)]
    for cohort in ('he_random', 'theorem_box'):
        f, tr = pairs(d, cohort)
        row = dict(pairs=len(f))
        for name, ids in (('fixed', f), ('trainable', tr)):
            y = np.log(d['gaps'][ids][:, win])
            alpha = -np.polyfit(x, y.T, 1)[0]
            local = []
            for a, b in zip(decs[:-1], decs[1:]):
                ga, gb = d['gaps'][ids, col(d, a)], d['gaps'][ids, col(d, b)]
                local.append(dict(interval=f'{a:.0e}-{b:.0e}', median_local_alpha=float(np.median(
                    -(np.log(gb)-np.log(ga))/(np.log(np.log(b))-np.log(np.log(a)))))))
            gl = d['gaps'][ids]*np.log(np.maximum(t, 2))
            row[name] = dict(
                alpha_fit_1e5_1e9=stats(alpha), alpha_median_ci95=boot_ci(alpha),
                local_alpha=local,
                gap=stats(d['gaps'][ids, col(d, 10**9)]),
                gap_by_decade={str(s): float(np.median(d['gaps'][ids, col(d, s)])) for s in decs},
                gap_times_lnt_ratio_1e9_over_1e5=stats(gl[:, col(d, 10**9)]/gl[:, col(d, 10**5)]),
                reduction_of_median_gap_1e7_to_1e9=float(1-np.median(d['gaps'][ids, col(d, 10**9)]) /
                                                         np.median(d['gaps'][ids, col(d, 10**7)])))
        out[cohort] = row

    mask = t >= 10
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for r, cohort in enumerate(('he_random', 'theorem_box')):
        f, tr = pairs(d, cohort)
        for m, ids in ((0, f), (1, tr)):
            g = d['gaps'][ids][:, mask]
            band(axes[r, 0], t[mask], g, MODE_COLOR[m], MODE_NAME[m])
            band(axes[r, 1], t[mask], g*np.log(t[mask]), MODE_COLOR[m], MODE_NAME[m])
        axes[r, 0].set_yscale('log')
        axes[r, 0].set_ylabel(f'{COHORT_NAME[cohort]}\nGap  1 − margin')
        axes[r, 1].set_ylabel('Gap × ln t')
        for ax in axes[r]:
            ax.set_xscale('log')
            ax.axvspan(10**5, 10**9, color='#f2f1ec', zorder=-1, lw=0)
        axes[r, 0].set_title(f'{len(f)} matched pairs', fontsize=10, color=INK)
        axes[r, 1].set_title('Flat = rate 1/ln t', fontsize=10, color=INK)
    for ax in axes[1]:
        ax.set_xlabel('GD updates')
    axes[0, 0].legend(loc='lower left', fontsize=9)
    fig.text(.5, -.01, 'Median and interquartile band, η = 0.05, 10⁹ updates. '
             'Shaded window 10⁵–10⁹: rate fit gap ≈ C/(ln t)^α.', ha='center', fontsize=8.5, color=MUTED)
    fig.tight_layout()
    save(fig, 'B_rate_gap_times_lnt')
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------- C
VARIANTS = [('unit', 'v = (1, −1)', ORANGE, '-'), ('balanced', 'v balanced, |v_j| = ‖(w_j, b_j)‖', AQUA, '--'),
            ('random', 'v ~ N(0, 1)', YELLOW, '-.'), ('small', 'v = 0.1·(1, −1)', MAGENTA, ':')]


def analyze_C(d, dA, plt):
    out = {}
    steps = [10**k for k in range(5, 8)]
    for cohort in ('he_random', 'theorem_box'):
        n = 2000 if cohort == 'he_random' else 32
        row = {}
        fA = select(dA, cohort, 0)
        fA = fA[dA['seed'][fA] < n]
        row['fixed_reference'] = dict(completed=len(fA), attempted=n,
                                      gap={str(s): stats(dA['gaps'][fA, col(dA, s)]) for s in steps})
        for v, *_ in VARIANTS:
            ids = select(d, cohort, 1, vinit=v)
            pT = d['params'][ids, -1]
            row[v] = dict(attempted=n, completed=len(ids),
                          gap={str(s): stats(d['gaps'][ids, col(d, s)]) for s in steps},
                          reduction_of_median_gap_1e5_to_1e7=float(
                              1-np.median(d['gaps'][ids, col(d, 10**7)])/np.median(d['gaps'][ids, col(d, 10**5)])),
                          v_sign_pattern_final={'+,-': int(np.sum((pT[:, 4] > 0) & (pT[:, 5] < 0))),
                                                'other': int(np.sum(~((pT[:, 4] > 0) & (pT[:, 5] < 0))))})
        out[cohort] = row

    t = d['checkpoints']
    mask = t >= 10
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, cohort in zip(axes, ('he_random', 'theorem_box')):
        n = 2000 if cohort == 'he_random' else 32
        fA = select(dA, cohort, 0)
        fA = fA[dA['seed'][fA] < n]
        curves = [(np.median(dA['gaps'][fA][:, mask], 0), 'Fixed output weights', BLUE, '-')]
        for v, label, color, ls in VARIANTS:
            ids = select(d, cohort, 1, vinit=v)
            curves.append((np.median(d['gaps'][ids][:, mask], 0), f'Trainable, {label}', color, ls))
        for y, label, color, ls in curves:
            ax.plot(t[mask], y, color=color, ls=ls, label=label)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('GD updates')
        ax.set_title(COHORT_NAME[cohort], fontsize=10, color=INK)
    axes[0].set_ylabel('Median gap  1 − margin')
    axes[1].legend(loc='center left', bbox_to_anchor=(1.02, .5), fontsize=8.5)
    fig.text(.45, -.03, 'Medians over completed runs of each variant (He: first 2,000 initialisations; box: 32). η = 0.05.',
             ha='center', fontsize=8.5, color=MUTED)
    fig.tight_layout()
    save(fig, 'C_output_weight_initialisation')
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------- D
def analyze_D(d, dA, plt):
    out = {}
    for cohort in ('he_random', 'theorem_box'):
        n = 2000 if cohort == 'he_random' else 32
        row = {}
        for lr, src in ((.01, d), (.05, dA), (.1, d)):
            f, tr = pairs(src, cohort, lr, seeds=set(range(n)))
            j5, j7 = col(src, 10**5), col(src, 10**7)
            row[str(lr)] = dict(pairs=len(f), fixed_gap_1e7=stats(src['gaps'][f, j7]),
                                trainable_gap_1e7=stats(src['gaps'][tr, j7]),
                                fixed_reduction_of_median_1e5_to_1e7=float(
                                    1-np.median(src['gaps'][f, j7])/np.median(src['gaps'][f, j5])),
                                trainable_reduction_of_median_1e5_to_1e7=float(
                                    1-np.median(src['gaps'][tr, j7])/np.median(src['gaps'][tr, j5])),
                                frac_trainable_larger=float(np.mean(src['gaps'][tr, j7] > src['gaps'][f, j7])))
        out[cohort] = row

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    lrs = [.01, .05, .1]
    for ax, cohort in zip(axes, ('he_random', 'theorem_box')):
        for k, (m, key) in enumerate(((0, 'fixed_gap_1e7'), (1, 'trainable_gap_1e7'))):
            xs = np.arange(3)+(k-.5)*.18
            s = [out[cohort][str(lr)][key] for lr in lrs]
            med = np.array([q['median'] for q in s])
            err = np.array([[q['median']-q['q25'] for q in s], [q['q75']-q['median'] for q in s]])
            ax.errorbar(xs, med, yerr=err, fmt='o', ms=6, color=MODE_COLOR[m], capsize=0, lw=1.5,
                        label=MODE_NAME[m])
        ax.set_xticks(range(3), [f'η = {lr:g}' for lr in lrs])
        ax.set_yscale('log')
        ax.grid(axis='x', visible=False)
        pr = out[cohort]['0.05']['pairs']
        ax.set_title(f'{COHORT_NAME[cohort]} (≈{pr:,} pairs per η)', fontsize=10, color=INK)
    axes[0].set_ylabel('Gap after 10⁷ updates')
    axes[1].legend(loc='center right', fontsize=8.5)
    fig.text(.5, -.03, 'Median with interquartile range over matched pairs.', ha='center', fontsize=8.5, color=MUTED)
    fig.tight_layout()
    save(fig, 'D_learning_rate')
    plt.close(fig)
    return out


def main():
    plt = setup()
    summary = {}
    dA = load('A')
    if dA is not None:
        summary['A_main'] = analyze_A(dA, plt)
        for exp, fn in (('B', analyze_B), ('C', analyze_C), ('D', analyze_D)):
            d = load(exp)
            if d is None:
                continue
            summary[{'B': 'B_rate', 'C': 'C_output_init', 'D': 'D_learning_rate'}[exp]] = (
                fn(d, plt) if exp == 'B' else fn(d, dA, plt))
    for exp in 'ABCD':
        m = ROOT/f'results_{exp}'/'meta.json'
        if m.exists():
            meta = json.loads(m.read_text())
            summary.setdefault('run_info', {})[exp] = {k: meta[k] for k in (
                'runs', 'completed', 'filter_failed', 'nonfinite', 'steps', 'train_seconds',
                'measure_seconds', 'workers')}
    (ROOT/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1)[:6000])


if __name__ == '__main__':
    main()

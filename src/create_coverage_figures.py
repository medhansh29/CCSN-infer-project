"""
Model-calibration figures: how often the observed photometry falls inside the
model's own 16th-84th percentile band, as a function of phase.

Reads the static light-curve payloads (data/static_payloads/*_lc.json), which
hold the observations and the latest model band in absolute magnitudes.
For each detected point inside the model's time grid:
    z = (obs - model_median) / sqrt(sigma_model^2 + sigma_obs^2)
with sigma_model = half-width of the 16-84 band. A calibrated model has
|z| < 1 for ~68% of points and |z| < 2 for ~95%.

Phase is days since the first model epoch (the fitted explosion time).
Standalone: python3 src/create_coverage_figures.py
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from create_paper_figures import PALETTE, set_paper_style, _save

PHASE_BINS = [0, 20, 50, 100, np.inf]
PHASE_LABELS = ['Rise\n0-20 d', 'Early plateau\n20-50 d', 'Plateau\n50-100 d', 'Tail\n>100 d']
BAND_STYLE = {'g': (PALETTE['green'], 'o', 'g'), 'r': (PALETTE['red'], 's', 'r')}
MIN_PTS_PER_CELL = 3          # points an event needs in a phase to get a misfit value
MIN_OBJ_PER_BIN = 5           # events a phase bin needs to be plotted in fig06
COVERAGE_EDGES = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 160])
EARLY_MAX, LATE_MIN = 20, 50  # fig07b: rise vs plateau+tail
CALIBRATED_MEDIAN_ABSZ = 0.674  # median |z| of a unit Gaussian


def load_points(payload_dir):
    rows = []
    for f in sorted(glob.glob(f"{payload_dir}/*_lc.json")):
        with open(f) as fh:
            d = json.load(fh)
        for o in d['observations']:
            mf = d['model_fit'].get(f"{o['filter']}_band")
            if mf is None or o['is_upperlimit'] or o['magerr'] is None:
                continue
            t = np.asarray(mf['mjd'])
            if not (t[0] <= o['mjd'] <= t[-1]):
                continue
            med = np.interp(o['mjd'], t, mf['median'])
            a = np.interp(o['mjd'], t, mf['upper_16th'])
            b = np.interp(o['mjd'], t, mf['lower_84th'])
            sig_m = abs(b - a) / 2
            rows.append(dict(obj=d['object_id'], band=o['filter'], phase=o['mjd'] - t[0],
                             resid=o['mag'] - med, sig_m=sig_m, sig_o=o['magerr'],
                             z=(o['mag'] - med) / np.hypot(sig_m, o['magerr'])))
    df = pd.DataFrame(rows)
    df['ph'] = pd.cut(df.phase, PHASE_BINS, labels=PHASE_LABELS, right=False)
    return df


# ---------------------------------------------------------------------------
# fig06: coverage vs phase
# ---------------------------------------------------------------------------
def _coverage_curve(d, edges, thresh, n_boot=1000, rng=None):
    """Object-weighted coverage per phase bin: each object's hit fraction counts
    once, so densely sampled objects don't dominate. 68% interval from a
    bootstrap over objects."""
    rng = rng or np.random.default_rng(0)
    centers, cov, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s = d[(d.phase >= a) & (d.phase < b)]
        frac = (s.z.abs() < thresh).groupby(s.obj).mean().to_numpy()
        if len(frac) < MIN_OBJ_PER_BIN:
            continue
        boots = frac[rng.integers(0, len(frac), (n_boot, len(frac)))].mean(axis=1)
        centers.append((a + b) / 2)
        cov.append(frac.mean())
        lo.append(np.percentile(boots, 16))
        hi.append(np.percentile(boots, 84))
    return map(np.asarray, (centers, cov, lo, hi))


def fig06_coverage_vs_phase(df, output_dir, objects=None, name='fig06_coverage_vs_phase'):
    d = df if objects is None else df[df.obj.isin(objects)]
    edges = COVERAGE_EDGES
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[3, 1], hspace=0.08))
    for col, (thresh, target) in enumerate(((1, 0.683), (2, 0.954))):
        ax = axes[0, col]
        for band, (color, marker, label) in BAND_STYLE.items():
            c, cov, lo, hi = _coverage_curve(d[d.band == band], edges, thresh)
            ax.fill_between(c, lo, hi, color=color, alpha=0.18, lw=0)
            ax.plot(c, cov, color=color, marker=marker, ms=3.5, lw=1.2, label=label)
        ax.axhline(target, color=PALETTE['black'], lw=0.8, ls='--')
        ax.text(edges[-1] - 2, target + 0.015, f'calibrated ({target:.0%})', ha='right', va='bottom', fontsize=6)
        ax.set_ylim(0, 1.02)
        ax.set_title(rf'Mean per-event fraction of points with $|z| < {thresh}$', fontsize=8)

        # events contributing to each bin
        axn = axes[1, col]
        w = np.diff(edges)
        for k, (band, (color, _, _)) in enumerate(BAND_STYLE.items()):
            s = d[d.band == band]
            n = [s[(s.phase >= a) & (s.phase < b)].obj.nunique() for a, b in zip(edges[:-1], edges[1:])]
            axn.bar(edges[:-1] + w * (0.1 + 0.4 * k), n, width=w * 0.38, align='edge', color=color, lw=0)
        axn.axhline(MIN_OBJ_PER_BIN, color=PALETTE['gray'], lw=0.6, ls='--')
        axn.set_xlabel('Phase (days since explosion)')
        for a in (ax, axn):
            for x in PHASE_BINS[1:-1]:
                a.axvline(x, color=PALETTE['gray'], lw=0.5, ls=':', zorder=0)
    axes[0, 0].set_ylabel('Coverage')
    axes[1, 0].set_ylabel('Events')
    axes[0, 1].set_yticklabels([])
    axes[1, 1].sharey(axes[1, 0])
    axes[1, 1].tick_params(labelleft=False)
    axes[0, 0].set_xlim(0, edges[-1])
    axes[0, 0].legend(loc='lower left', title='Band', title_fontsize=7)
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout()
    _save(fig, output_dir, name)


# ---------------------------------------------------------------------------
# fig07: per-event misfit, one dot per event
# ---------------------------------------------------------------------------
def per_event_misfit(df):
    """Median |z| per (event, phase), kept only where the event has enough points."""
    g = df.groupby(['obj', 'ph'], observed=True).z
    t = pd.DataFrame({'med_absz': g.apply(lambda z: z.abs().median()), 'n': g.size()}).reset_index()
    return t[t.n >= MIN_PTS_PER_CELL]


def _median_absz(df, mask):
    s = df[mask].groupby('obj').z
    t = pd.DataFrame({'m': s.apply(lambda z: z.abs().median()), 'n': s.size()})
    return t.m[t.n >= MIN_PTS_PER_CELL]


def fig07_misfit_by_event(df, output_dir, n_label=3):
    t = per_event_misfit(df)
    rng = np.random.default_rng(1)
    fig, (ax, axs) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw=dict(width_ratios=[1.35, 1]))

    # (a) distribution of per-event misfit in each phase
    for i, ph in enumerate(PHASE_LABELS):
        s = t[t.ph == ph].sort_values('med_absz', ascending=False)
        if s.empty:
            continue
        x = i + rng.uniform(-0.22, 0.22, len(s))
        ax.scatter(x, s.med_absz, s=9, color=PALETTE['blue'], alpha=0.55, lw=0, zorder=2)
        q1, q2, q3 = np.percentile(s.med_absz, [25, 50, 75])
        ax.add_patch(plt.Rectangle((i - 0.32, q1), 0.64, q3 - q1, fill=False, ec=PALETTE['black'], lw=0.8, zorder=3))
        ax.plot([i - 0.32, i + 0.32], [q2, q2], color=PALETTE['black'], lw=1.4, zorder=3)
        # worst events labelled beside the column, nudged apart in log-y
        ly = np.log10(s.med_absz.to_numpy()[:n_label])
        for k in range(1, len(ly)):
            ly[k] = min(ly[k], ly[k - 1] - 0.09)
        for (_, r), xi, yl in zip(s.head(n_label).iterrows(), x[:n_label], ly):
            ax.annotate(r.obj.replace('ZTF', ''), (xi, r.med_absz), xytext=(i + 0.3, 10 ** yl),
                        fontsize=5, va='center', color=PALETTE['black'],
                        arrowprops=dict(arrowstyle='-', lw=0.4, color=PALETTE['gray'], shrinkA=0, shrinkB=1.5))
        ax.text(i, 0.055, f'N={len(s)}', ha='center', va='bottom', fontsize=6, color=PALETTE['gray'])
    ax.axhline(CALIBRATED_MEDIAN_ABSZ, color=PALETTE['black'], lw=0.8, ls='--')
    ax.text(len(PHASE_LABELS) - 0.5, CALIBRATED_MEDIAN_ABSZ * 0.92, 'calibrated', ha='right', va='top', fontsize=6)
    ax.set_yscale('log')
    ax.set_ylim(0.05, 12)
    ax.set_yticks([0.1, 0.3, 1, 3, 10], ['0.1', '0.3', '1', '3', '10'])
    ax.set_xticks(range(len(PHASE_LABELS)), PHASE_LABELS, fontsize=6.5)
    ax.set_xlim(-0.5, len(PHASE_LABELS) - 0.5)
    ax.set_ylabel(r'Per-event median $|z|$')
    ax.set_title('(a) Misfit per event and phase', fontsize=8, loc='left')

    # (b) early vs late misfit for events observed in both
    early = _median_absz(df, df.phase < EARLY_MAX)
    late = _median_absz(df, df.phase >= LATE_MIN)
    both = pd.concat({'early': early, 'late': late}, axis=1).dropna()
    c = CALIBRATED_MEDIAN_ABSZ
    lim = (0.05, 12)
    axs.axvspan(2 * c, lim[1], color=PALETTE['orange'], alpha=0.10, lw=0)
    axs.axvline(c, color=PALETTE['black'], lw=0.8, ls='--')
    axs.axhline(c, color=PALETTE['black'], lw=0.8, ls='--')
    axs.plot(lim, lim, color=PALETTE['gray'], lw=0.6, ls=':')
    axs.scatter(both.early, both.late, s=12, color=PALETTE['blue'], alpha=0.7, lw=0, zorder=2)
    for oid, r in both[both.early > 2 * c].iterrows():
        axs.annotate(oid.replace('ZTF', ''), (r.early, r.late), xytext=(3, 2), textcoords='offset points', fontsize=5)
    axs.text(lim[1] * 0.9, lim[0] * 1.15, 'early fit\n> 2× calibrated', ha='right', va='bottom', fontsize=6, color=PALETTE['black'])
    axs.set_xscale('log'); axs.set_yscale('log')
    axs.set_xlim(lim); axs.set_ylim(lim)
    for a in (axs.xaxis, axs.yaxis):
        a.set_major_locator(plt.FixedLocator([0.1, 0.3, 1, 3, 10]))
        a.set_major_formatter(plt.FixedFormatter(['0.1', '0.3', '1', '3', '10']))
    axs.set_xlabel(rf'Early median $|z|$ (< {EARLY_MAX} d)')
    axs.set_ylabel(rf'Late median $|z|$ ($\geq$ {LATE_MIN} d)')
    axs.set_title(f'(b) Events with early and late data (N={len(both)})', fontsize=8, loc='left')
    fig.tight_layout()
    _save(fig, output_dir, 'fig07_misfit_by_event')
    return t, both


# ---------------------------------------------------------------------------
# fig08: residuals vs phase with model-uncertainty envelope
# ---------------------------------------------------------------------------
def _rolling(x, y, centers, half_width):
    out = []
    for c in centers:
        m = np.abs(x - c) <= half_width
        out.append(np.median(y[m]) if m.sum() >= 10 else np.nan)
    return np.asarray(out)


def fig08_residuals_vs_phase(df, output_dir, ylim=2.5):
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.2), sharex=True, sharey=True)
    centers = np.arange(2, 160, 2.0)
    for ax, (band, (color, marker, label)) in zip(axes, BAND_STYLE.items()):
        d = df[df.band == band]
        x, r = d.phase.to_numpy(), d.resid.to_numpy()
        env = _rolling(x, d.sig_m.to_numpy(), centers, 8)
        ax.fill_between(centers, -env, env, color=PALETTE['gray'], alpha=0.35, lw=0,
                        label=r'model 1$\sigma$ (16-84%)')
        clipped = np.clip(r, -ylim, ylim)
        out = np.abs(r) > ylim
        ax.scatter(x[~out], clipped[~out], s=4, color=color, alpha=0.35, lw=0, marker=marker)
        ax.scatter(x[out], clipped[out], s=10, color=color, alpha=0.6, marker='x', lw=0.6)
        ax.plot(centers, _rolling(x, r, centers, 8), color=PALETTE['black'], lw=1.2,
                label='running median residual')
        ax.axhline(0, color=PALETTE['black'], lw=0.6)
        for xb in PHASE_BINS[1:-1]:
            ax.axvline(xb, color=PALETTE['gray'], lw=0.5, ls=':', zorder=0)
        ax.text(0.01, 0.95, f'{label} band  (N = {len(d)} pts, {d.obj.nunique()} objects)',
                transform=ax.transAxes, ha='left', va='top', fontsize=7)
        ax.set_ylabel(r'$m_{\rm obs} - m_{\rm model}$ (mag)')
        ax.set_ylim(ylim * 1.05, -ylim * 1.05)  # magnitudes: brighter (negative) up
    axes[0].text(1.0, 1.0, 'obs. brighter than model ↑', transform=axes[0].transAxes,
                 ha='right', va='bottom', fontsize=6, color=PALETTE['gray'])
    axes[0].legend(loc='lower right', ncol=2, fontsize=6)
    axes[1].set_xlabel('Phase (days since explosion)')
    axes[1].set_xlim(0, 160)
    fig.tight_layout()
    _save(fig, output_dir, 'fig08_residuals_vs_phase')


# ---------------------------------------------------------------------------
def create_coverage_figures(payload_dir='data/static_payloads', output_dir='data/paper_figures'):
    set_paper_style()
    os.makedirs(output_dir, exist_ok=True)
    df = load_points(payload_dir)
    df.to_csv(f'{output_dir}/coverage_points.csv', index=False)
    fig06_coverage_vs_phase(df, output_dir)
    per_phase, early_late = fig07_misfit_by_event(df, output_dir)
    per_phase.to_csv(f'{output_dir}/coverage_median_absz_by_phase.csv', index=False)
    early_late.to_csv(f'{output_dir}/coverage_early_vs_late.csv', index_label='obj')
    fig08_residuals_vs_phase(df, output_dir)
    return df


if __name__ == "__main__":
    create_coverage_figures()

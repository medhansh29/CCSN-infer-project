"""
Publication-quality figures restyled from existing pipeline output.

Reads only the CSVs produced by main.py (convergence / uncertainty /
scatter_outliers) and computes nothing new beyond plotting-level summaries
(Pearson r of the *_final columns, histogram bins, KDE contours, OLS line).
Standalone: see generate_paper_figures.py at the repo root.
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# Okabe-Ito colorblind-safe palette
PALETTE = {
    'blue':   '#0072B2',
    'orange': '#E69F00',
    'green':  '#009E73',
    'red':    '#D55E00',
    'purple': '#CC79A7',
    'yellow': '#F0E442',
    'gray':   '#999999',
    'black':  '#000000',
}

PARAMS = ['zams', 'mloss_rate', '56Ni', 'k_energy', 'beta', 'texp', 'A_v', 'logZ']

# Short labels (mathtext so they render with any font)
SHORT_LABELS = {
    'zams': 'ZAMS', 'mloss_rate': r'$\dot{M}$', '56Ni': r'$^{56}$Ni',
    'k_energy': r'$E_k$', 'beta': r'$\beta$', 'texp': r'$t_{\mathrm{exp}}$',
    'A_v': r'$A_V$', 'logZ': r'$\log Z$',
}

# Axis labels with units (same conventions as create_summary_plots.py)
AXIS_LABELS = {
    'zams': r'ZAMS (M$_\odot$)',
    'mloss_rate': r'$-\log_{10}\dot{M}$ (M$_\odot$ yr$^{-1}$)',
    '56Ni': r'$^{56}$Ni (M$_\odot$)',
    'k_energy': r'$E_k$ ($10^{51}$ erg)',
    'beta': r'$\beta$',
    'texp': r'$t_{\mathrm{exp}}$ (day)',
    'A_v': r'$A_V$ (mag)',
    'logZ': r'$\log Z$',
}

BIVARIATE = {
    'Mloss-Ek':  ('mloss_rate', 'k_energy'),
    'Ek-Ni':     ('k_energy', '56Ni'),
    'Texp-Beta': ('texp', 'beta'),
    'logZ-Av':   ('logZ', 'A_v'),
}

# Parameter triples per cluster, mirroring multivariate_outliers.py
CLUSTERS = {
    'Energy Engine':        ['k_energy', 'mloss_rate', '56Ni'],
    'Progenitor Evolution': ['zams', 'mloss_rate', 'logZ'],
    'Modeling Degeneracy':  ['A_v', 'texp', 'logZ'],
    'Ejecta Efficiency':    ['k_energy', 'mloss_rate', 'beta'],
    'LC Morphology':        ['texp', '56Ni', 'A_v'],
}

# Direction -> (color, marker). Marker shape carries the meaning in grayscale.
DIRECTION_STYLE = {
    'Above Trendline': (PALETTE['orange'], '^'),
    'Below Trendline': (PALETTE['red'], 'v'),
}

MIN_POINTS_FOR_CONTOURS = 50
N_LABELED_BIVARIATE = 3


def set_paper_style():
    mpl.rcParams.update({
        "figure.figsize": (3.4, 2.6),
        "font.size": 8,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 7,
        "lines.linewidth": 1.2,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save(fig, output_dir, name):
    fig.savefig(f"{output_dir}/{name}.pdf")
    fig.savefig(f"{output_dir}/{name}.png", dpi=300)
    plt.close(fig)


def _fmt_r(v, pos=None):
    """Pearson r with the leading zero dropped: 0.42 -> .42, -0.42 -> -.42"""
    s = f"{v:.2f}"
    return s.replace("0.", ".", 1) if abs(v) < 1 else s


def _outlier_style(direction):
    return DIRECTION_STYLE.get(direction, (PALETTE['orange'], '^'))


# ---------------------------------------------------------------------------
# fig01
# ---------------------------------------------------------------------------
def fig01_correlation_matrix(conv, output_dir):
    cols = [f'{p}_final' for p in PARAMS]
    corr = conv[cols].corr(method='pearson').values
    n = len(PARAMS)
    masked = np.ma.masked_where(np.triu(np.ones((n, n), dtype=bool)), corr)

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('white', 0)
    im = ax.imshow(masked, cmap=cmap, vmin=-1, vmax=1)

    for i in range(n):
        for j in range(i):
            v = corr[i, j]
            ax.text(j, i, _fmt_r(v), ha='center', va='center', fontsize=6,
                    color='white' if abs(v) > 0.6 else 'black')

    labels = [SHORT_LABELS[p] for p in PARAMS]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_yticklabels(labels)
    ax.set_xlim(-0.5, n - 1.5)
    ax.set_ylim(n - 0.5, 0.5)
    ax.spines[['left', 'bottom']].set_visible(False)
    ax.tick_params(length=2)

    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, shrink=0.8)
    cb.set_label('Pearson r')
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=7, width=0.5)
    _save(fig, output_dir, 'fig01_correlation_matrix')


# ---------------------------------------------------------------------------
# fig02
# ---------------------------------------------------------------------------
def fig02_uncertainty_distributions(unc, output_dir):
    cols = [f'{p}_rel_uncertainty' for p in PARAMS]
    pooled = unc[cols].to_numpy().ravel()
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size < 2:
        print("⚠️  fig02: no finite uncertainty values; skipping")
        return False

    lo, hi = pooled.min(), pooled.max()
    q75, q25 = np.percentile(pooled, [75, 25])
    iqr = q75 - q25
    width = 2 * iqr / pooled.size ** (1 / 3) if iqr > 0 else (hi - lo) / 10
    n_bins = int(np.clip(np.ceil((hi - lo) / width), 5, 60)) if width > 0 else 10
    edges = np.linspace(lo, hi, n_bins + 1)

    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.6), sharex=True)
    for ax, p in zip(axes.ravel(), PARAMS):
        v = unc[f'{p}_rel_uncertainty'].dropna().to_numpy()
        ax.hist(v, bins=edges, histtype='step', color=PALETTE['black'], lw=1.0)
        ax.axvline(0.20, color=PALETTE['blue'], lw=0.8, ls='--')
        ax.axvline(0.50, color=PALETTE['red'], lw=0.8, ls='--')
        if v.size:
            ax.axvline(np.median(v), color=PALETTE['gray'], lw=1.0, ls=':')
            # fractions per band, printed as small text (readable without a table)
            f = [np.mean(v < 0.2), np.mean((v >= 0.2) & (v < 0.5)), np.mean(v >= 0.5)]
            ax.text(0.97, 0.97, f"{f[0]:.0%} | {f[1]:.0%} | {f[2]:.0%}",
                    transform=ax.transAxes, ha='right', va='top', fontsize=6)
        ax.set_xlim(lo, hi)
        ax.set_xlabel(SHORT_LABELS[p] + ' rel. unc.', fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel('Objects')
    fig.tight_layout()
    _save(fig, output_dir, 'fig02_uncertainty_distributions')
    return True


# ---------------------------------------------------------------------------
# fig03
# ---------------------------------------------------------------------------
def fig03_bivariate_outliers(conv, outliers, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4))
    for ax, (otype, (xp, yp)) in zip(axes.ravel(), BIVARIATE.items()):
        xc, yc = f'{xp}_final', f'{yp}_final'
        d = conv[[xc, yc]].dropna()
        ax.scatter(d[xc], d[yc], s=8, alpha=0.35, color=PALETTE['gray'],
                   linewidths=0, zorder=1)
        if len(d) >= 2 and d[xc].nunique() > 1:
            slope, icpt = np.polyfit(d[xc], d[yc], 1)
            xs = np.array([d[xc].min(), d[xc].max()])
            ax.plot(xs, slope * xs + icpt, color=PALETTE['blue'], lw=0.9, zorder=2)

        sub = outliers[outliers['outlier_type'] == otype]
        for direction, (color, marker) in DIRECTION_STYLE.items():
            s = sub[sub['direction'] == direction]
            ax.scatter(s['x_param_value'], s['y_param_value'], s=22, marker=marker,
                       color=color, edgecolors='black', linewidths=0.4, zorder=3,
                       label=direction.replace(' Trendline', ''))
        top = sub.sort_values('distance_from_trend', ascending=False).head(N_LABELED_BIVARIATE)
        xlo, xhi = ax.get_xlim()
        for rank, (_, r) in enumerate(top.iterrows()):
            # stagger vertically by rank so nearby points don't collide;
            # right-align labels for points in the right part of the axes
            right = (r['x_param_value'] - xlo) / (xhi - xlo) > 0.6
            ax.annotate(r['object_id'], (r['x_param_value'], r['y_param_value']),
                        xytext=(-4 if right else 4, 4 - 9 * rank),
                        ha='right' if right else 'left',
                        textcoords='offset points', fontsize=5.5)
        ax.set_xlabel(AXIS_LABELS[xp])
        ax.set_ylabel(AXIS_LABELS[yp])
        if ax is axes[0, 0]:
            ax.legend(loc='best', handletextpad=0.2)
    fig.tight_layout()
    _save(fig, output_dir, 'fig03_bivariate_outliers')


# ---------------------------------------------------------------------------
# fig04
# ---------------------------------------------------------------------------
def _kde_contours(ax, x, y, color):
    """68%/95% highest-density contours from a Gaussian KDE."""
    try:
        kde = gaussian_kde(np.vstack([x, y]))
    except (np.linalg.LinAlgError, ValueError):
        return False
    px, py = np.ptp(x) * 0.15, np.ptp(y) * 0.15
    gx = np.linspace(x.min() - px, x.max() + px, 100)
    gy = np.linspace(y.min() - py, y.max() + py, 100)
    X, Y = np.meshgrid(gx, gy)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    order = np.sort(Z.ravel())[::-1]
    cum = np.cumsum(order)
    cum /= cum[-1]
    lv95, lv68 = order[np.searchsorted(cum, 0.95)], order[np.searchsorted(cum, 0.68)]
    ax.contour(X, Y, Z, levels=[lv95, lv68], colors=[color], linewidths=[0.6, 1.0])
    return True


def fig04_corner_clusters(conv, outliers, output_dir):
    """Returns {cluster: used_contours(bool)} for caption generation."""
    used = {}
    for cluster, params in CLUSTERS.items():
        cols = [f'{p}_final' for p in params]
        d = conv[['object_id'] + cols].dropna()
        use_contours = len(d) > MIN_POINTS_FOR_CONTOURS
        used[cluster] = use_contours
        flagged_ids = outliers.loc[outliers['outlier_type'] == cluster, 'object_id'].unique()
        flagged = d[d['object_id'].isin(flagged_ids)]
        dirs = outliers[outliers['outlier_type'] == cluster].set_index('object_id')['direction']

        k = len(params)
        fig, axes = plt.subplots(k, k, figsize=(6.5, 6.5))
        for i in range(k):
            for j in range(k):
                ax = axes[i, j]
                if j > i:
                    ax.axis('off')
                    continue
                pj, pi = params[j], params[i]
                if i == j:
                    v = d[cols[i]].to_numpy()
                    ax.hist(v, bins='fd' if len(v) > 1 else 1, histtype='step',
                            color=PALETTE['black'], lw=1.0)
                    med, sd = np.median(v), np.std(v)
                    ax.set_title(f"{med:.2f} ± {sd:.2f}", fontsize=7)
                    ax.set_yticks([])
                    ax.spines['left'].set_visible(False)
                else:
                    x, y = d[cols[j]].to_numpy(), d[cols[i]].to_numpy()
                    ok = use_contours and _kde_contours(ax, x, y, PALETTE['blue'])
                    if not ok:
                        ax.scatter(x, y, s=6, alpha=0.4, color=PALETTE['blue'], linewidths=0)
                    for _, r in flagged.iterrows():
                        color, marker = _outlier_style(dirs.get(r['object_id']))
                        ax.scatter(r[cols[j]], r[cols[i]], s=20, marker=marker, color=color,
                                   edgecolors='black', linewidths=0.4, zorder=3)
                        ax.annotate(r['object_id'], (r[cols[j]], r[cols[i]]),
                                    xytext=(3, 3), textcoords='offset points', fontsize=5)
                if i == k - 1:
                    ax.set_xlabel(AXIS_LABELS[pj])
                else:
                    ax.set_xticklabels([])
                if j == 0 and i > 0:
                    ax.set_ylabel(AXIS_LABELS[pi])
                elif i != j:
                    ax.set_yticklabels([])
        fig.tight_layout()
        slug = cluster.lower().replace(' ', '_')
        _save(fig, output_dir, f'fig04_corner_{slug}')
    return used


# ---------------------------------------------------------------------------
# fig05
# ---------------------------------------------------------------------------
def _agreement_counts(conv, outliers):
    ids = conv['object_id']
    morph = conv.set_index('object_id')['is_anomaly'].astype(bool)
    biv = set(outliers.loc[outliers['outlier_type'].isin(BIVARIATE), 'object_id'])
    mult = set(outliers.loc[outliers['outlier_type'].isin(CLUSTERS), 'object_id'])
    return pd.Series({o: int(morph[o]) + int(o in biv) + int(o in mult) for o in ids})


def fig05_detector_agreement(conv, outliers, output_dir):
    counts = _agreement_counts(conv, outliers).value_counts().reindex(range(4), fill_value=0)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    # Sequential blue ramp: monotone in luminance, so it survives grayscale
    shades = ['#C6DBEF', '#6BAED6', '#2171B5', '#08306B']
    bars = ax.bar(counts.index, counts.values, color=shades, edgecolor='black', linewidth=0.5)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha='center', va='bottom', fontsize=7)
    ax.set_xticks(range(4))
    ax.set_xlabel('Number of detectors flagging object')
    ax.set_ylabel('Objects')
    ax.margins(y=0.12)
    _save(fig, output_dir, 'fig05_detector_agreement')
    return counts


# ---------------------------------------------------------------------------
# captions
# ---------------------------------------------------------------------------
def _write_captions(output_dir, n_obj, contour_used, agree, fig02_made):
    bands = "0.20 and 0.50 (dashed vertical lines) separate well-, moderately-, and weakly-constrained parameters; the dotted line marks the sample median; the inset text gives the fraction of objects in each band"
    small = [c for c, u in contour_used.items() if not u]
    scatter_note = (f" With N = {n_obj} objects, panels for {', '.join(small)} show alpha-blended "
                    "scatter rather than 68%/95% contours because the sample is too small for "
                    "reliable density estimation.") if small else ""
    multi = int((agree >= 2).sum())
    lines = ["# Suggested captions", ""]
    lines += ["## fig01_correlation_matrix", "",
              f"Pearson correlation matrix of the final inferred parameters for N = {n_obj} objects. "
              "The upper triangle and diagonal are omitted. Color scale is fixed to [-1, 1].", ""]
    if fig02_made:
        lines += ["## fig02_uncertainty_distributions", "",
                  f"Distributions of relative posterior uncertainty for each of the eight parameters. {bands[0].upper() + bands[1:]}.", ""]
    lines += ["## fig03_bivariate_outliers", "",
              "Bivariate parameter relations. Gray points: full population; blue line: OLS trend. "
              "Flagged outliers are marked with orange upward triangles (above trend) and red downward "
              "triangles (below trend); the three most extreme per panel are labeled by object ID.", ""]
    for c, u in contour_used.items():
        slug = c.lower().replace(' ', '_')
        lines += [f"## fig04_corner_{slug}", "",
                  f"Population-level corner plot for the {c} parameter triplet. Diagonal panels show "
                  "marginal histograms with median ± 1σ of the population. Off-diagonal panels show 68% and 95% "
                  "density contours. Multivariate (Mahalanobis) outliers are marked as in Fig. 3 and labeled "
                  f"by object ID.{scatter_note}", ""]
    lines += ["## fig05_detector_agreement", "",
              "Number of independent anomaly detectors flagging each object: (1) Isolation Forest on "
              "light-curve morphological features, (2) bivariate physical-parameter outliers, and "
              "(3) multivariate Mahalanobis outliers. Detector (1) operates on morphology while (2) and (3) "
              f"operate on inferred physical parameters, so agreement is not circular. {multi} of {n_obj} "
              "objects are flagged by two or more detectors.", ""]
    with open(f"{output_dir}/captions.md", "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
def create_paper_figures(convergence_file, uncertainty_file, outliers_file,
                         output_dir='data/paper_figures'):
    set_paper_style()
    os.makedirs(output_dir, exist_ok=True)
    conv = pd.read_csv(convergence_file)
    unc = pd.read_csv(uncertainty_file)
    outliers = pd.read_csv(outliers_file)

    fig01_correlation_matrix(conv, output_dir)
    fig02_made = fig02_uncertainty_distributions(unc, output_dir)
    fig03_bivariate_outliers(conv, outliers, output_dir)
    contour_used = fig04_corner_clusters(conv, outliers, output_dir)
    fig05_detector_agreement(conv, outliers, output_dir)
    agree = _agreement_counts(conv, outliers)
    _write_captions(output_dir, len(conv), contour_used, agree, fig02_made)


if __name__ == "__main__":
    create_paper_figures('data/convergence_metrics.csv', 'data/uncertainty_metrics.csv',
                         'data/scatter_outliers.csv')

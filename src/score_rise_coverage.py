"""
Rise-time calibration score per object: how often the observed rise photometry
falls inside the model's own 16th-84th percentile band.

Only the rise is scored; everything after it is ignored. The rise of each band
runs from the first model epoch (the fitted explosion time) until the model
median first comes within PEAK_TOL mag of its brightest value, or for a fixed
number of days with --rise-days. (The brightest model point itself is no good
as the rise end: IIP plateaus often keep brightening slowly, pushing it deep
into the plateau.) For each detected rise point:
    z = (obs - model_median) / sqrt(sigma_model^2 + sigma_obs^2)
with sigma_model = half-width of the 16-84 band. A calibrated model has
|z| < 1 for ~68% of points and |z| < 2 for ~95%.

Detections before the first model epoch can't be scored, but they mean the
model exploded too late, so they are counted separately (n_pre_texp).

Every object with at least one rise point is scored and can be flagged; scores
from fewer than MIN_RISE_PTS points are kept but noted as sparse. Objects with
no rise detections are listed with a note.

Candidates are flagged objects that also have late-time data (>= MIN_LATE_PTS
detections from LATE_MIN days on): the late light curve should have pinned the
fit down, yet the rise is still badly fit. Each candidate gets a plot of its
full light curve against the model band (--fig-dir).

Standalone: python3 src/score_rise_coverage.py [--payload-dir DIR] [--rise-days N]
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from create_paper_figures import PALETTE, set_paper_style, _save

PEAK_TOL = 0.2                  # mag below model peak that ends the rise
MIN_RISE_PTS = 3                # fewer rise points than this -> score noted as sparse
LATE_MIN = 50                   # days after fitted explosion where late time starts
MIN_LATE_PTS = 3                # late detections an object needs to count as having late data
CALIBRATED_MEDIAN_ABSZ = 0.674  # median |z| of a unit Gaussian


def load_rise_points(payload_dir, rise_days=None):
    rows, info = [], {}
    for f in sorted(glob.glob(f"{payload_dir}/*_lc.json")):
        with open(f) as fh:
            d = json.load(fh)
        oid = d['object_id']
        info[oid] = dict(n_pre_texp=0, n_late=0, first_det_d=np.inf, rise_len_d=0.0)
        for band in ('g', 'r'):
            mf = d['model_fit'].get(f"{band}_band")
            if mf is None:
                continue
            t = np.asarray(mf['mjd'])
            med_curve = np.asarray(mf['median'])
            t_end = t[0] + rise_days if rise_days else t[np.argmax(med_curve <= med_curve.min() + PEAK_TOL)]
            info[oid]['rise_len_d'] = max(info[oid]['rise_len_d'], t_end - t[0])
            for o in d['observations']:
                if o['filter'] != band or o['is_upperlimit'] or o['magerr'] is None:
                    continue
                info[oid]['first_det_d'] = min(info[oid]['first_det_d'], o['mjd'] - t[0])
                if o['mjd'] < t[0]:
                    info[oid]['n_pre_texp'] += 1
                    continue
                if o['mjd'] - t[0] >= LATE_MIN:
                    info[oid]['n_late'] += 1
                if o['mjd'] > t_end:
                    continue
                med = np.interp(o['mjd'], t, med_curve)
                a = np.interp(o['mjd'], t, mf['upper_16th'])
                b = np.interp(o['mjd'], t, mf['lower_84th'])
                sig_m = abs(b - a) / 2
                rows.append(dict(obj=oid, band=band, phase=o['mjd'] - t[0], resid=o['mag'] - med,
                                 sig_m=sig_m, sig_o=o['magerr'],
                                 z=(o['mag'] - med) / np.hypot(sig_m, o['magerr'])))
    return pd.DataFrame(rows), pd.DataFrame(info).T


def _scores(z):
    a = z.abs()
    return pd.Series({'n_rise': len(z), 'cov1': (a < 1).mean(), 'cov2': (a < 2).mean(),
                      'med_absz': a.median(), 'bias': z.mean()})


def score_objects(df, info):
    """One row per object, worst rise fit first. Positive bias = observed
    fainter than the model (model too bright / rises too early)."""
    s = df.groupby('obj').z.apply(_scores).unstack()
    for band in ('g', 'r'):
        b = df[df.band == band].groupby('obj').z.apply(_scores).unstack()
        s = s.join(b[['n_rise', 'cov1', 'med_absz']].add_suffix(f'_{band}'))
    s = s.join(info, how='outer')
    s['n_pre_texp'] = s.n_pre_texp.astype(int)
    s['n_late'] = s.n_late.astype(int)
    s['n_rise'] = s.n_rise.fillna(0).astype(int)
    s['flag'] = (s.med_absz > 2 * CALIBRATED_MEDIAN_ABSZ) | (s.cov1 < 0.5) | (s.n_pre_texp > 0)
    s['candidate'] = s.flag & (s.n_rise > 0) & (s.n_late >= MIN_LATE_PTS)
    s['note'] = [_note(r) for r in s.itertuples()]
    return s.sort_values(['candidate', 'flag', 'med_absz'], ascending=[False, False, False])


def _note(r):
    notes = []
    if r.n_rise == 0:
        notes.append(f'no rise detections: first detection {r.first_det_d:.0f} d after fitted explosion, '
                     f'model rise ends {r.rise_len_d:.0f} d (found post-peak, or explosion fitted too early)')
    elif r.n_rise < MIN_RISE_PTS:
        notes.append(f'sparse rise ({r.n_rise} pt{"s" if r.n_rise > 1 else ""}): low-confidence score')
    if r.n_late < MIN_LATE_PTS:
        notes.append(f'little late-time data ({r.n_late} detections >= {LATE_MIN} d)')
    if r.n_pre_texp:
        notes.append(f'{r.n_pre_texp} detection(s) before fitted explosion')
    return '; '.join(notes)


# ---------------------------------------------------------------------------
# per-object rise plots
# ---------------------------------------------------------------------------
Z_MAX = 4  # |z| colour scale saturates here
Z_CMAP = plt.get_cmap('Reds')


def _zcolor(z):
    return Z_CMAP(0.3 + 0.7 * min(abs(z), Z_MAX) / Z_MAX)


def plot_rise(payload_file, row, rise_days, output_dir):
    with open(payload_file) as fh:
        d = json.load(fh)
    bands = [b for b in ('g', 'r') if f"{b}_band" in d['model_fit']]
    fig, axes = plt.subplots(1, len(bands), figsize=(3.7 * len(bands), 2.8), squeeze=False, sharey=True)
    for ax, band in zip(axes[0], bands):
        mf = d['model_fit'][f"{band}_band"]
        t = np.asarray(mf['mjd'])
        med = np.asarray(mf['median'])
        t_end = t[0] + rise_days if rise_days else t[np.argmax(med <= med.min() + PEAK_TOL)]
        rise_len = t_end - t[0]
        obs = [o for o in d['observations'] if o['filter'] == band]
        det = [o for o in obs if not o['is_upperlimit'] and o['magerr'] is not None]
        x_lo = min([-5] + [o['mjd'] - t[0] - 3 for o in det if o['mjd'] < t[0]])
        x_hi = max(t[-1] - t[0], max([o['mjd'] - t[0] for o in obs], default=0)) + 5
        ph = t - t[0]
        ax.axvspan(0, rise_len, color=PALETTE['gray'], alpha=0.12, lw=0)
        ax.fill_between(ph, mf['upper_16th'], mf['lower_84th'], color=PALETTE['blue'], alpha=0.25, lw=0,
                        label=r'model 1$\sigma$ (16-84%)')
        ax.plot(ph, med, color=PALETTE['blue'], lw=1.2, label='model median')
        ax.axvline(0, color=PALETTE['black'], lw=0.8, ls='--')
        for o in obs:
            x = o['mjd'] - t[0]
            if not (x_lo <= x <= x_hi):
                continue
            if o['is_upperlimit'] or o['magerr'] is None:
                ax.plot(x, o['mag'], 'v', ms=4, mfc='none', mec=PALETTE['gray'], mew=0.7)
            elif x < 0:
                ax.errorbar(x, o['mag'], o['magerr'], fmt='X', ms=6, color=PALETTE['black'], lw=0.7, zorder=4)
            elif x > rise_len:  # past the rise: shown, not scored
                ax.errorbar(x, o['mag'], o['magerr'], fmt='o', ms=4, mfc='none', color=PALETTE['gray'], lw=0.6)
            else:
                m = np.interp(o['mjd'], t, med)
                sig = np.hypot(abs(np.interp(o['mjd'], t, mf['lower_84th']) - np.interp(o['mjd'], t, mf['upper_16th'])) / 2,
                               o['magerr'])
                ax.errorbar(x, o['mag'], o['magerr'], fmt='o', ms=5, color=_zcolor((o['mag'] - m) / sig),
                            mec=PALETTE['black'], mew=0.5, ecolor=PALETTE['black'], elinewidth=0.6, zorder=5)
        ax.set_xlim(x_lo, x_hi)
        ax.set_xlabel('Days since fitted explosion')
        ax.set_title(f'{band} band', fontsize=8, loc='left')
    # y-range from the photometry: the model starts near 0 mag at explosion,
    # which would otherwise squash the data into a thin strip
    shown = [o['mag'] for o in d['observations'] if o['filter'] in bands]
    ax0 = axes[0][0]
    ax0.set_ylim(max(shown) + 1.5, min(shown) - 1.0)
    ax0.set_ylabel('Absolute mag')
    ax0.legend(fontsize=6, loc='lower right', frameon=False)
    sm = plt.cm.ScalarMappable(cmap=plt.matplotlib.colors.ListedColormap(Z_CMAP(np.linspace(0.3, 1, 256))),
                               norm=plt.Normalize(0, Z_MAX))
    cax = fig.add_axes([0.915, 0.17, 0.012, 0.6])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(r'rise point $|z|$', fontsize=7)
    head = (f"{row.Index}   rise pts {row.n_rise}   within 1$\\sigma$ {row.cov1:.2f}   "
            f"median |z| {row.med_absz:.2f}   bias {row.bias:+.2f}   late pts {row.n_late}")
    fig.suptitle(head + (f"\n{row.note}" if row.note else ''), fontsize=8, x=0.02, ha='left')
    fig.text(0.99, 0.01, 'shaded: rise window   X: detection before explosion   '
             'hollow: after rise (not scored)   v: upper limit', fontsize=5.5, ha='right', color=PALETTE['gray'])
    fig.subplots_adjust(left=0.08, right=0.89, bottom=0.17, top=0.8 if row.note else 0.86, wspace=0.08)
    _save(fig, output_dir, f"rise_{row.Index}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--payload-dir', default='data/static_payloads')
    p.add_argument('--rise-days', type=float, default=None,
                   help='fixed rise window in days after explosion (default: up to model peak)')
    p.add_argument('--out', default='data/rise_coverage.csv')
    p.add_argument('--fig-dir', default='data/rise_figures')
    args = p.parse_args()

    df, info = load_rise_points(args.payload_dir, args.rise_days)
    s = score_objects(df, info)
    s.to_csv(args.out, float_format='%.3f')
    n_sparse = int(((s.n_rise > 0) & (s.n_rise < MIN_RISE_PTS)).sum())
    print(f"{len(s)} objects: {int((s.n_rise >= MIN_RISE_PTS).sum())} well-sampled rises, "
          f"{n_sparse} sparse (<{MIN_RISE_PTS} pts), {int((s.n_rise == 0).sum())} with no rise data; "
          f"{int(s.flag.sum())} flagged, {int(s.candidate.sum())} of them with late-time data (candidates)")
    print(f"pooled rise coverage: |z|<1 {(df.z.abs() < 1).mean():.2f} (target 0.68), "
          f"|z|<2 {(df.z.abs() < 2).mean():.2f} (target 0.95)")
    cols = ['n_rise', 'n_late', 'cov1', 'med_absz', 'bias', 'n_pre_texp', 'first_det_d', 'flag', 'candidate', 'note']
    with pd.option_context('display.width', 250, 'display.max_rows', 200, 'display.max_colwidth', 80):
        print(s[cols].round(2).to_string())

    os.makedirs(args.fig_dir, exist_ok=True)
    set_paper_style()
    for row in s[s.candidate].itertuples():
        plot_rise(f"{args.payload_dir}/{row.Index}_lc.json", row, args.rise_days, args.fig_dir)
    print(f"{int(s.candidate.sum())} plots -> {args.fig_dir}")
    print(f"-> {args.out}")


if __name__ == '__main__':
    main()

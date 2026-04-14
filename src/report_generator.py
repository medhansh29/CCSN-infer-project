import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import subprocess
import shutil

def escape_latex(s):
    if not isinstance(s, str): return s
    return s.replace('_', r'\_').replace('%', r'\%').replace('&', r'\&').replace('$', r'\$')

def ensure_model_plot(oid, base_dir='.'):
    """Copy the latest model_absolute_nn plot for an OID into data/report_images/ and return its absolute file URI."""
    dates = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('202')])
    img_dir = os.path.join(base_dir, 'data', 'report_images')
    for d in reversed(dates):
        ap = os.path.join(base_dir, d, f"{oid}_model_absolute_nn.png")
        if os.path.exists(ap):
            sub_img_dir = f"{oid}_{d}"
            full_target_dir = os.path.join(img_dir, sub_img_dir)
            os.makedirs(full_target_dir, exist_ok=True)
            target_path = os.path.join(full_target_dir, f"{oid}_lc.png")
            if not os.path.exists(target_path):
                shutil.copy(ap, target_path)
            # Return absolute path so the PDF hyperlink works on macOS
            abs_path = os.path.abspath(target_path).replace(' ', '%20')
            return abs_path
    return None

def generate_report():
    try:
        df = pd.read_csv('data/convergence_metrics.csv')
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return

    # Load External Template
    template_path = os.path.join(os.path.dirname(__file__), 'templates', 'report_template.tex')
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            report_template = f.read()
    except Exception as e:
        print(f"Error loading template: {e}")
        return

    filtered_rows = ""
    try:
        fdf = pd.read_csv('data/flagged_non_iip_objects.csv')
        f_list = []
        for _, row in fdf.iterrows():
            oid = row['object_id']
            reason = str(row.get('flag_reason', 'N/A')).replace('_', r'\_')
            f_list.append(rf"{escape_latex(oid)} & \alerce{{{oid}}} & Filtered ({escape_latex(reason)}) \\")
        filtered_rows = "\n".join(f_list)
    except:
        filtered_rows = r"N/A & N/A & N/A \\"

    scatter_outliers_rows = ""
    multivariate_outliers_rows = ""
    try:
        odf = pd.read_csv('data/scatter_outliers.csv')
        
        clusters = ["Energy Engine", "Progenitor Evolution", "Modeling Degeneracy", 
                    "Ejecta Efficiency", "LC Morphology"]
                    
        # Multi-Variate (3D) Processing
        odf_3d = odf[odf['outlier_type'].isin(clusters)].copy()
        if not odf_3d.empty:
            # Take top 3 per cluster, then group by OID to merge duplicates
            odf_3d = odf_3d.sort_values(by='distance_from_trend', ascending=False)
            top_odf_3d = odf_3d.groupby('outlier_type').head(3).copy()
            
            # Group by object_id: merge types, keep max distance
            grouped_3d = top_odf_3d.groupby('object_id').agg(
                types=('outlier_type', lambda x: ', '.join(sorted(x.unique()))),
                max_dist=('distance_from_trend', 'max'),
                n_types=('outlier_type', 'nunique')
            ).reset_index().sort_values('max_dist', ascending=False)
            
            m_list = []
            for _, grow in grouped_3d.iterrows():
                oid = grow['object_id']
                otype = grow['types']
                dist = float(grow['max_dist'])
                rcolor = r"\rowcolor{mutered} " if grow['n_types'] > 1 else ""
                plot_path = ensure_model_plot(oid)
                link_str = rf"\href{{file:///{plot_path}}}{{View Plot}}" if plot_path else "N/A"
                m_list.append(rf"{rcolor}\alerce{{{escape_latex(oid)}}} & {otype} & {dist:.2f} & {grow['n_types']} clusters & {link_str} \\")
            multivariate_outliers_rows = "\n".join(m_list)
        else:
            multivariate_outliers_rows = r"N/A & N/A & N/A & N/A & N/A \\"
            
        # Bi-Variate (2D) Processing
        odf_2d = odf[~odf['outlier_type'].isin(clusters)].copy()
        if not odf_2d.empty:
            # Take top 3 per combo, then group by OID
            odf_2d = odf_2d.sort_values(by='distance_from_trend', ascending=False)
            top_odf_2d = odf_2d.groupby('outlier_type').head(3).copy()
            
            # Group by object_id: merge types, keep max distance
            grouped_2d = top_odf_2d.groupby('object_id').agg(
                types=('outlier_type', lambda x: ', '.join(sorted(x.unique()))),
                max_dist=('distance_from_trend', 'max'),
                n_types=('outlier_type', 'nunique'),
                direction=('direction', 'first')
            ).reset_index().sort_values('max_dist', ascending=False)
            
            o_list = []
            for _, grow in grouped_2d.iterrows():
                oid = grow['object_id']
                otype = grow['types'].replace('_', r'\_')
                direction = str(grow['direction'])
                dist = float(grow['max_dist'])
                rcolor = r"\rowcolor{mutered} " if grow['n_types'] > 1 else ""
                plot_path = ensure_model_plot(oid)
                link_str = rf"\href{{file:///{plot_path}}}{{View Plot}}" if plot_path else "N/A"
                o_list.append(rf"{rcolor}\alerce{{{escape_latex(oid)}}} & {otype} & {direction} & {dist:.3f} & {link_str} \\")
            scatter_outliers_rows = "\n".join(o_list)
        else:
            scatter_outliers_rows = r"N/A & N/A & N/A & N/A & N/A \\"
            
    except Exception as e:
        print(f"Error loading scatter outliers: {e}")
        scatter_outliers_rows = r"N/A & N/A & N/A & N/A \\"
        multivariate_outliers_rows = r"N/A & N/A & N/A & N/A \\"

    df = df[df['fit_success'].isin([True, 'True'])]
    THRESHOLD = 5 
    if 'num_observations' in df.columns:
        reliable_df = df[df['num_observations'] >= THRESHOLD].copy()
    else:
        reliable_df = df.copy()
    
    if reliable_df.empty:
        print("No reliable objects to report.")
        return

    # Image Shadow Directory
    img_dir = 'data/report_images'
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    os.makedirs(img_dir, exist_ok=True)

    cat1_list = []
    cat2_list = []
    cat3_list = []
    cat4_list = []
    c_bs_list = []
    c_env_list = []
    coupled_profiles = ""

    for _, row in reliable_df.iterrows():
        oid = row['object_id']
        
        m_obs = row.get('M_plateau_25d', np.nan)
        m_exp = row.get('M_peak_predicted', np.nan)
        m_diff = row.get('M_peak_residual', 0)
        cat1_lum = abs(m_diff) > 0.75
        cat1_budget = str(row.get('mass_budget_violation', '')) in ['True', 'true']
        ni = row.get('56Ni_final', 0)
        mej = max(row.get('implied_Mej', 10), 1)
        cat1_ni = (ni/mej > 0.01)

        t_obs = row.get('plateau_duration_days', np.nan)
        t_exp = row.get('plateau_duration_predicted', np.nan)
        t_diff = row.get('plateau_duration_residual', 0)
        cat2_ext = abs(t_diff) > 20

        prec = str(row.get('precursor_flag', '')) in ['True', 'true', '1', '1.0']
        rise = str(row.get('early_rise_excess_flag', '')) in ['True', 'true', '1', '1.0']
        cool = str(row.get('arrested_cooling_flag', '')) in ['True', 'true', '1', '1.0']

        reb = str(row.get('rebrightening_flag', '')) in ['True', 'true', '1', '1.0']
        lin = str(row.get('linear_residual_flag', '')) in ['True', 'true', '1', '1.0']

        primary_cat = None
        if cat1_lum or cat1_budget or cat1_ni: primary_cat = "I"
        elif cat2_ext: primary_cat = "II"
        elif prec or rise or cool: primary_cat = "III"
        elif reb or lin: primary_cat = "IV"

        if cat1_lum or cat1_budget or cat1_ni:
            flags = []
            if cat1_lum: flags.append(rf"\textcolor{{outlierred}}{{Luminosity Excess ({m_diff:+.2f})}}")
            if cat1_budget: flags.append(r"\textcolor{warningorange}{Mass Budget ($E_k/M_{ej}>1$)}")
            if cat1_ni: flags.append(r"\textcolor{warningorange}{Nickel Overabundance (>1\%)}")
            z_ek = row.get('zams_final', 15) / max(row.get('k_energy_final', 1), 0.1)
            cat1_list.append(rf"\hyperref[sec:{oid}]{{{escape_latex(oid)}}} & {m_obs:.2f} & {m_exp:.2f} & {z_ek:.2f} & {' + '.join(flags)} \\")

        if cat2_ext:
            cat2_list.append(rf"\hyperref[sec:{oid}]{{{escape_latex(oid)}}} & {t_obs:.1f} & {t_exp:.1f} & \textcolor{{outlierred}}{{{t_diff:+.1f} days}} \\")

        if prec or rise or cool:
            p_s = r"\textcolor{outlierred}{Detected}" if prec else "None"
            r_s = r"\textcolor{warningorange}{Excess}" if rise else "Normal"
            c_s = r"\textcolor{warningorange}{Arrested}" if cool else "Normal"
            prim = "Precursor" if prec else ("CSM Breakout" if rise else "CSM Interaction")
            cat3_list.append(rf"\hyperref[sec:{oid}]{{{escape_latex(oid)}}} & {p_s} & {r_s} & {c_s} & {prim} \\")

        if reb or lin:
            r_s = r"\textcolor{outlierred}{Yes}" if reb else "No"
            l_s = r"\textcolor{warningorange}{Detected}" if lin else "No"
            cat4_list.append(rf"\hyperref[sec:{oid}]{{{escape_latex(oid)}}} & {r_s} & {l_s} \\")

        p_reasons = []
        action_items = []
        if cat1_lum: 
            p_reasons.append(f"Luminosity Excess ({m_diff:+.2f} mag)")
            action_items.append("Action Required: Verify the absolute magnitude scaling. A significant deviation (>0.75 mag) may indicate an incorrect redshift or a non-standard explosion energy.")
        if cat1_budget: 
            p_reasons.append("Mass Budget Violation")
            action_items.append("Action Required: Check the $E_k/M_{ej}$ ratio. Values >1.0 are unphysical for standard IIP models; verify data quality or consider magnetar/CSM powering.")
        if cat1_ni: 
            p_reasons.append(f"Nickel Overabundance ({(ni/mej)*100:.1f}%)")
            action_items.append("Action Required: Inspect the late-time slope. A high Nickel fraction (>1%) suggests a massive progenitor or a possible transition to a peculiar type.")
        if cat2_ext: 
            p_reasons.append(f"Plateau Deviation ({t_diff:+.1f} days)")
            action_items.append("Action Required: Examine the plateau duration. Extreme deviations (>20 days) suggest an anomalous hydrogen envelope mass or a misidentified transition to the radioactive tail.")
        if prec: 
            p_reasons.append("Precursor Detected")
            action_items.append("Action Required: Inspect the pre-explosion baseline. The pipeline flagged a >3\\sigma precursor; manually verify against background noise or host-galaxy artifacts.")
        if rise: 
            p_reasons.append("Early Rise Excess")
            action_items.append("Action Required: Check the rise phase. A >0.1 mag excess above the fireball curve indicates shock breakout interaction with dense CSM.")
        if cool: 
            p_reasons.append("Arrested Cooling")
            action_items.append("Action Required: Review the early color evolution. Artificially blue gradients suggest sustained heating from early-time interaction.")
        if reb: 
            p_reasons.append("Rebrightening Bump")
            action_items.append("Action Required: Check the mid-plateau photometry (Days 30-40). The model struggled to fit a detected rebrightening bump; verify this is real signal.")
        if lin: 
            p_reasons.append("Linear Residual Cluster")
            action_items.append("Action Required: Examine the plateau residuals. A contiguous cluster of deviations indicates a non-linear cooling phase or a hidden CSM shock.")

        if cat1_lum and cat2_ext:
            c_bs_list.append(rf"\alerce{{{oid}}} & {m_diff:+.2f} mag & {t_diff:+.1f} days & \hyperref[sec:{oid}]{{See Profile}} \\")
        if prec and (rise or cool):
            c_env_list.append(rf"\alerce{{{oid}}} & Detected & Verified & \hyperref[sec:{oid}]{{See Profile}} \\")

        if p_reasons:
            l_abs = None
            l_corner = None
            run_folder = "Unknown"
            sub_img_dir = None
            
            dates = sorted([d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('202')])
            for d in reversed(dates):
                ap = os.path.join(d, f"{oid}_model_absolute_nn.png")
                cp = os.path.join(d, f"{oid}_corner_plot.jpg")
                if (not l_abs and os.path.exists(ap)) or (not l_corner and os.path.exists(cp)):
                    run_folder = d
                    sub_img_dir = f"{oid}_{d}"
                    full_target_dir = os.path.join(img_dir, sub_img_dir)
                    os.makedirs(full_target_dir, exist_ok=True)
                    
                    if not l_abs and os.path.exists(ap):
                        shutil.copy(ap, os.path.join(full_target_dir, f"{oid}_lc.png"))
                        l_abs = f"report_images/{sub_img_dir}/{oid}_lc.png"
                    if not l_corner and os.path.exists(cp):
                        shutil.copy(cp, os.path.join(full_target_dir, f"{oid}_corner.jpg"))
                        l_corner = f"report_images/{sub_img_dir}/{oid}_corner.jpg"
                    
                    jsons = glob.glob(os.path.join(d, f"{oid}*.json"))
                    for js in jsons:
                        shutil.copy(js, os.path.join(full_target_dir, os.path.basename(js)))
                
                if l_abs and l_corner: break
            
            profile = rf"\clearpage" + "\n" + rf"\subsection[{escape_latex(oid)}]{{\alerce{{{oid}}}}} \label{{sec:{oid}}}" + "\n"
            profile += rf"\textbf{{Processing Run:}} {escape_latex(run_folder)} \quad | \quad \hyperref[sec:cat{primary_cat}]{{[Back to Ledger {primary_cat}]}}" + "\\\\\n\n"
            profile += rf"\textbf{{Flag Details:}} {', '.join(p_reasons)}\\" + "\n"
            
            profile += r"\textbf{Required Review Action(s):}" + "\n" + r"\begin{itemize}" + "\n"
            for item in action_items:
                profile += rf"    \item {item}" + "\n"
            profile += r"\end{itemize}" + "\n\n"
            
            fig_str = r"\vspace{0.5cm}" + "\n" + r"\begin{figure}[H]" + "\n" + r"\centering" + "\n"
            if l_abs:
                fig_str += rf"\begin{{minipage}}{{0.48\textwidth}}" + "\n" + r"\centering" + "\n" + rf"\includegraphics[width=\linewidth]{{{l_abs}}}" + "\n" + r"\caption*{Best-Fit Light Curve}" + "\n" + r"\end{minipage}\hfill" + "\n"
            if l_corner:
                fig_str += rf"\begin{{minipage}}{{0.48\textwidth}}" + "\n" + r"\centering" + "\n" + rf"\includegraphics[width=\linewidth]{{{l_corner}}}" + "\n" + r"\caption*{Posterior Corner Plot}" + "\n" + r"\end{minipage}" + "\n"
            fig_str += r"\end{figure}" + "\n" + r"\vspace{0.5cm}" + "\n" + r"\hrule" + "\n" + r"\vspace{0.5cm}" + "\n"
            coupled_profiles += profile + fig_str

    report_tex = report_template.replace("{run_date}", datetime.now().strftime("%Y-%m-%d"))
    report_tex = report_tex.replace("{total_objects}", str(len(reliable_df)))
    report_tex = report_tex.replace('{flagged_limit_rows}', filtered_rows)
    report_tex = report_tex.replace('{scatter_outliers_rows}', scatter_outliers_rows if scatter_outliers_rows else r"N/A & N/A & N/A & N/A \\")
    report_tex = report_tex.replace('{multivariate_outliers_rows}', multivariate_outliers_rows if multivariate_outliers_rows else r"N/A & N/A & N/A & N/A \\")
    report_tex = report_tex.replace("{cat1_rows}", "\n".join(cat1_list) if cat1_list else r"N/A & N/A & N/A & N/A & N/A \\")
    report_tex = report_tex.replace("{cat2_rows}", "\n".join(cat2_list) if cat2_list else r"N/A & N/A & N/A & N/A \\")
    report_tex = report_tex.replace("{cat3_rows}", "\n".join(cat3_list) if cat3_list else r"N/A & N/A & N/A & N/A & N/A \\")
    report_tex = report_tex.replace("{cat4_rows}", "\n".join(cat4_list) if cat4_list else r"N/A & N/A & N/A \\")
    report_tex = report_tex.replace("{coupled_bs_rows}", "\n".join(c_bs_list) if c_bs_list else r"N/A & N/A & N/A & N/A \\")
    report_tex = report_tex.replace("{coupled_env_rows}", "\n".join(c_env_list) if c_env_list else r"N/A & N/A & N/A & N/A \\")
    report_tex = report_tex.replace("{coupled_profiles}", coupled_profiles if coupled_profiles else r"\textit{No flagged objects found.}")
    report_tex = report_tex.replace("{filtered_rows}", filtered_rows)

    os.makedirs('data', exist_ok=True)
    tex_path = 'data/diagnostic_report.tex'
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(report_tex)
        
    print(f"✅ Generated LaTeX report at {tex_path}")
    print("⏳ Compiling PDF...")
    try:
        for _ in range(2):
            subprocess.run(['pdflatex', '-interaction=nonstopmode', 'diagnostic_report.tex'], 
                           cwd='data', capture_output=True)
        print("✅ Successfully compiled to data/diagnostic_report.pdf")
    except Exception as e:
        print(f"⚠️ Could not compile PDF: {e}")

if __name__ == "__main__":
    generate_report()

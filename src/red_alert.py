#!/usr/bin/env python3
"""
Red Alert Module

Analyzes final inferred parameters and uncertainty metrics to flag objects
that are highly suspicious. Flags include:
1. High confidence (low uncertainty) but poor evidence or high volatility.
2. Unphysical parameter constraints (e.g. Explosion Energy vs Ejecta Mass).
"""

import pandas as pd
import numpy as np
import argparse
import os

def generate_red_alerts(
    convergence_file: str = 'data/convergence_metrics.csv',
    uncertainty_file: str = 'data/uncertainty_metrics.csv',
    output_file: str = 'data/red_alerts.csv'
):
    """
    Ingest metrics and generate red alerts for suspicious inferences.
    """
    if not os.path.exists(convergence_file) or not os.path.exists(uncertainty_file):
        print(f"⚠️ Missing input metrics CSVs. Cannot run Red Alert module.")
        return

    conv_df = pd.read_csv(convergence_file)
    unc_df = pd.read_csv(uncertainty_file)

    # Merge on object_id to have all features in one dataframe
    df = pd.merge(conv_df, unc_df, on='object_id', suffixes=('', '_unc'))
    
    alerts = []

    for _, row in df.iterrows():
        obj_id = row['object_id']
        obj_alerts = []
        
        # 1. High Confidence but Volatile / Poor Evidence (Model is "Confident but wrong")
        # We check across all main parameters for low uncertainty (<10%) paired with high volatility
        for param in ['zams', 'mloss_rate', '56Ni', 'k_energy', 'beta', 'texp', 'A_v']:
            rel_unc_col = f'{param}_rel_uncertainty'
            vol_col = f'{param}_volatility_std'
            
            if rel_unc_col in row and vol_col in row:
                if pd.notna(row[rel_unc_col]) and pd.notna(row[vol_col]):
                    # If uncertainty is very tight (<5%) but trajectory standard deviation was high
                    # zams > 2 M_sun variance, Ni > 0.05 variance, Ek > 1 FOE variance
                    is_volatile = False
                    if param == 'zams' and row[vol_col] > 2.0: is_volatile = True
                    elif param == '56Ni' and row[vol_col] > 0.05: is_volatile = True
                    elif param == 'k_energy' and row[vol_col] > 1.0: is_volatile = True
                    elif row[vol_col] > 0.5: is_volatile = True # Generic fallback
                    
                    if row[rel_unc_col] < 0.05 and is_volatile:
                        obj_alerts.append(f'Narrow posterior (<5% unc) but highly volatile {param} trajectory (std={row[vol_col]:.2f})')
        
        # Check against log evidence
        if 'log_evidence' in row and pd.notna(row['log_evidence']):
            # If evidence is exceptionally poor (-200 is a very weak fit for refitt, depending on dataset)
            if row['log_evidence'] < -200:
                obj_alerts.append(f'Extremely poor log evidence ({row["log_evidence"]:.1f}) indicating failed global fit')
                
        # 2. Unphysical / Extreme Physical Constraints
        # Explosion Energy vs Ejecta Mass
        zams, ek, ni = None, None, None
        if 'zams_final' in row: zams = row['zams_final']
        if 'k_energy_final' in row: ek = row['k_energy_final']
        if '56Ni_final' in row: ni = row['56Ni_final']
        
        if pd.notna(zams) and pd.notna(ek):
            mej_est = max(0.1, zams - 1.5)  # Rest mass minus neutron star remnant
            ratio = ek / mej_est
            if ratio > 1.5:  # e.g. 5 FOE for 3 M_sun ejecta is unphysical for standard IIP
                obj_alerts.append(f'Unphysical Energy/Mass ratio: E_k ({ek:.1f} FOE) is too high for estimated M_ej ({mej_est:.1f} M_sun)')
            if ek > 6.0:
                obj_alerts.append(f'Extreme explosion energy (>6 FOE): {ek:.2f}')
            if ek < 0.1:
                obj_alerts.append(f'Abnormally low explosion energy (<0.1 FOE): {ek:.2f}')
                
        # Check extreme Nickel masses
        if pd.notna(ni):
            if ni > 0.3:
                obj_alerts.append(f'Unphysically high 56Ni mass for SN IIP (>0.3 M_sun): {ni:.3f}')
            if pd.notna(zams) and ni > 0.05 * zams:
                obj_alerts.append(f'56Ni mass ({ni:.3f}) is suspiciously large fraction of ZAMS ({zams:.1f})')
                
        # 3. Asymmetric Posterior with high confidence
        for param in ['zams', 'k_energy']:
            asym_col = f'{param}_asymmetry_index'
            unc_col = f'{param}_rel_uncertainty'
            if asym_col in row and unc_col in row:
                if pd.notna(row[asym_col]) and pd.notna(row[unc_col]):
                    if row[asym_col] > 0.6 and row[unc_col] < 0.1:
                        obj_alerts.append(f'High confidence (<10% unc) but highly asymmetric {param} posterior (skew > 0.6)')
        
        # Removed: 4. Partial Lightcurve Warn
        # (Completeness status is now tracked as metadata instead of a red alert trigger)
        
        # If any alerts triggered, record them
        if obj_alerts:
            alerts.append({
                'object_id': obj_id,
                'flag_count': len(obj_alerts),
                'alert_reasons': ' | '.join(obj_alerts),
                'completeness_status': row.get('completeness_status', 'Unknown'),
                'zams_final': row.get('zams_final', np.nan),
                'k_energy_final': row.get('k_energy_final', np.nan),
                '56Ni_final': row.get('56Ni_final', np.nan),
                'log_evidence': row.get('log_evidence', np.nan)
            })

    alerts_df = pd.DataFrame(alerts)
    
    if not alerts_df.empty:
        # Sort by flag count (most severe first)
        alerts_df = alerts_df.sort_values(by='flag_count', ascending=False)
        alerts_df.to_csv(output_file, index=False)
        print(f"🚨 Red Alert Module found {len(alerts_df)} suspicious objects.")
        print(f"💾 Saved red alerts to {output_file}")
    else:
        print("✅ Red Alert Module ran cleanly: No suspicious objects flagged.")
        # Create an empty CSV so the rest of the pipeline knows it ran
        pd.DataFrame(columns=[
            'object_id', 'flag_count', 'alert_reasons',
            'zams_final', 'k_energy_final', '56Ni_final', 'log_evidence'
        ]).to_csv(output_file, index=False)

    return alerts_df


def generate_relative_percentile_alerts(df: pd.DataFrame, alerts_list: list):
    """
    Flags the bottom/top 5% of the dataset for key metrics.
    """
    if len(df) < 10:
        return alerts_list # Not enough data for meaningful percentiles
        
    # 1. Bottom 5% of log evidence
    if 'log_evidence' in df.columns:
        p05_evidence = df['log_evidence'].quantile(0.05)
        worst_evidence = df[df['log_evidence'] <= p05_evidence]
        for _, row in worst_evidence.iterrows():
            alerts_list.append({
                'object_id': row['object_id'],
                'alert_type': 'Relative Anomaly',
                'alert_reason': f"Bottom 5% of Log Evidence ({row['log_evidence']:.1f})",
                'severity_score': 1,
                'completeness_status': row.get('completeness_status', 'Unknown'),
                'zams_final': row.get('zams_final', np.nan),
                'k_energy_final': row.get('k_energy_final', np.nan),
                '56Ni_final': row.get('56Ni_final', np.nan),
                'log_evidence': row.get('log_evidence', np.nan)
            })
            
    # 2. Top 5% most highly skewed (asymmetric) posteriors
    asym_cols = [c for c in df.columns if 'asymmetry_index' in c]
    if asym_cols:
        # Create a max asymmetry column across all parameters
        df['max_asymmetry'] = df[asym_cols].max(axis=1)
        p95_asym = df['max_asymmetry'].quantile(0.95)
        skewed = df[df['max_asymmetry'] >= p95_asym]
        for _, row in skewed.iterrows():
            worst_param = df[asym_cols].loc[row.name].idxmax().replace('_asymmetry_index', '')
            alerts_list.append({
                'object_id': row['object_id'],
                'alert_type': 'Relative Anomaly',
                'alert_reason': f"Top 5% most asymmetric posterior ({worst_param} skew={row['max_asymmetry']:.2f})",
                'severity_score': 1,
                'completeness_status': row.get('completeness_status', 'Unknown'),
                'zams_final': row.get('zams_final', np.nan),
                'k_energy_final': row.get('k_energy_final', np.nan),
                '56Ni_final': row.get('56Ni_final', np.nan),
                'log_evidence': row.get('log_evidence', np.nan)
            })
            
    return alerts_list


def generate_ml_anomalies(df: pd.DataFrame, alerts_list: list):
    """
    Uses an Isolation Forest to find multi-dimensional anomalies in the 7 parameter space.
    Calculates z-scores to explain the driving factors of the anomaly.
    """
    from sklearn.ensemble import IsolationForest
    from scipy.stats import zscore
    
    # Define the 7D parameter space to analyze
    features = [
        'zams_final', 'k_energy_final', '56Ni_final', 
        'mloss_rate_final', 'beta_final', 'texp_final', 'A_v_final'
    ]
    
    # Drop rows that haven't generated final parameters yet
    ml_df = df.dropna(subset=features).copy()
    
    if len(ml_df) < 10:
        return alerts_list
        
    # Fit AI (Contamination=0.1 means flag the 10% weirdest objects)
    clf = IsolationForest(contamination=0.1, random_state=42)
    ml_df['anomaly_score'] = clf.fit_predict(ml_df[features])
    
    # Filter to only the anomalies (-1)
    anomalies = ml_df[ml_df['anomaly_score'] == -1].copy()
    
    if not anomalies.empty:
        # Calculate z-scores for the whole population to find the "driving" feature
        z_df = ml_df[features].apply(zscore)
        
        for idx, row in anomalies.iterrows():
            # Find the parameter with the most extreme deviation (max absolute z-score)
            z_scores_abs = z_df.loc[idx].abs()
            top_feature = z_scores_abs.idxmax()
            top_z = z_df.loc[idx, top_feature]
            
            # Format feature name for readability
            feature_name = top_feature.replace('_final', '')
            direction = "High" if top_z > 0 else "Low"
            
            reason = f"ML Multi-dimensional Anomaly: Driven strongly by {direction} {feature_name} (Z={top_z:.1f})"
            
            alerts_list.append({
                'object_id': row['object_id'],
                'alert_type': 'Machine Learning Anomaly',
                'alert_reason': reason,
                'severity_score': 2,
                'completeness_status': row.get('completeness_status', 'Unknown'),
                'zams_final': row.get('zams_final', np.nan),
                'k_energy_final': row.get('k_energy_final', np.nan),
                '56Ni_final': row.get('56Ni_final', np.nan),
                'log_evidence': row.get('log_evidence', np.nan)
            })

    return alerts_list


def main():
    parser = argparse.ArgumentParser(description="Flag suspicious pipeline inferences.")
    parser.add_argument('--convergence', default='data/convergence_metrics.csv')
    parser.add_argument('--uncertainties', default='data/uncertainty_metrics.csv')
    parser.add_argument('--alerts-output', default='data/red_alerts.csv')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.convergence) or not os.path.exists(args.uncertainties):
        print(f"⚠️ Missing input metrics CSVs. Cannot run Anomaly modules.")
        return

    conv_df = pd.read_csv(args.convergence)
    unc_df = pd.read_csv(args.uncertainties)
    df = pd.merge(conv_df, unc_df, on='object_id', suffixes=('', '_unc'))
    
    print("\n--- Running Integrated Red Alert & Anomaly Engine ---")
    
    # Keep track of all consolidated alerts
    all_alerts = []
    
    # 1. Physics & Logic Red Alerts 
    # (Extracts alerts straight from df iterrows to match dictionary structure)
    physics_alerts = generate_red_alerts(
        convergence_file=args.convergence,
        uncertainty_file=args.uncertainties,
        output_file='/dev/null' # Ignore inner saving mechanism
    )
    if not physics_alerts.empty:
        for _, row in physics_alerts.iterrows():
            all_alerts.append({
                'object_id': row['object_id'],
                'alert_type': 'Physics / Validation Flag',
                'alert_reason': row['alert_reasons'],
                'severity_score': 3,  # Physics violations are highly severe
                'completeness_status': row.get('completeness_status', 'Unknown'),
                'zams_final': row.get('zams_final', np.nan),
                'k_energy_final': row.get('k_energy_final', np.nan),
                '56Ni_final': row.get('56Ni_final', np.nan),
                'log_evidence': row.get('log_evidence', np.nan)
            })
    
    # 2. Relative Percentile Alerts
    all_alerts = generate_relative_percentile_alerts(df, all_alerts)
    
    # 3. ML Isolation Forest Alerts
    all_alerts = generate_ml_anomalies(df, all_alerts)
    
    # Save EVERYTHING sequentially
    if all_alerts:
        out_df = pd.DataFrame(all_alerts)
        
        # Sort by severity score and then object ID
        out_df = out_df.sort_values(by=['severity_score', 'object_id'], ascending=[False, True])
        
        # Clean up column order
        cols = ['object_id', 'alert_type', 'alert_reason', 'severity_score', 'completeness_status', 
                'zams_final', 'k_energy_final', '56Ni_final', 'log_evidence']
        out_df = out_df[[c for c in cols if c in out_df.columns]]
        out_df.to_csv(args.alerts_output, index=False)
        print(f"\n🚨 Discovered {len(out_df)} total anomalies/flags across all modules.")
        print(f"💾 Consolidated Red Alerts + ML Anomalies saved to {args.alerts_output}")
    else:
        print("\n✅ Engine ran cleanly: No suspicious objects or anomalies flagged.")
        pd.DataFrame(columns=['object_id', 'alert_type', 'alert_reason', 'severity_score', 
                              'completeness_status', 'zams_final', 'k_energy_final', 
                              '56Ni_final', 'log_evidence']).to_csv(args.alerts_output, index=False)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Create summary visualizations from batch convergence analysis results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import os


def create_summary_plots(metrics_file: str, output_dir: str):
    """
    Create summary visualizations from convergence metrics.
    
    Args:
        metrics_file: Path to convergence metrics CSV
        output_dir: Directory to save plots
    """
    # Load data
    try:
        df = pd.read_csv(metrics_file)
        
        # Merge uncertainty data if available so that plots dependent on it trigger
        unc_df_path = os.path.join(os.path.dirname(metrics_file), 'uncertainty_metrics.csv')
        if not os.path.exists(unc_df_path) and os.path.exists('data/uncertainty_metrics.csv'):
            unc_df_path = 'data/uncertainty_metrics.csv'
            
        if os.path.exists(unc_df_path):
            unc_df = pd.read_csv(unc_df_path)
            # Only merge columns that don't already exist to avoid _x/_y suffixes
            merge_cols = ['object_id'] + [c for c in unc_df.columns if c not in df.columns]
            df = df.merge(unc_df[merge_cols], on='object_id', how='left')
            
    except FileNotFoundError:
        print(f"Error: {metrics_file} not found. Run batch_analyze_objects.py first.")
        return

    Path(output_dir).mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Define parameters and titles
    params = ['zams', 'mloss_rate', '56Ni', 'k_energy', 'beta', 'texp', 'A_v']
    titles = ['ZAMS (Mass)', 'Mass-Loss Rate', '56Ni Mass', 
              'Explosion Energy', 'Beta', 'Explosion Time', 'Extinction A_V']
    
    # ========================================================================
    # 1. N_90 Distribution Plot
    # ========================================================================
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (param, title) in enumerate(zip(params, titles)):
        ax = axes[idx]
        col_name = f'{param}_n90_days'
        
        if col_name in df.columns:
            data = df[col_name].dropna()
            
            if len(data) > 0:
                ax.hist(data, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
                ax.axvline(data.mean(), color='red', linestyle='--', 
                           label=f'Mean: {data.mean():.1f} d', linewidth=2)
                ax.axvline(data.median(), color='orange', linestyle='--',
                           label=f'Median: {data.median():.1f} d', linewidth=2)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
        
        ax.set_xlabel('Days to 10% Convergence (N_90)', fontsize=10)
        ax.set_ylabel('Number of Objects', fontsize=10)
        ax.set_title(f'{title}\nConvergence Time', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(len(params), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/n90_distributions.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/n90_distributions.png")
    plt.close()
    
    # ========================================================================
    # 2. Volatility vs N_90 Scatter
    # ========================================================================
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (param, title) in enumerate(zip(params, titles)):
        ax = axes[idx]
        vol_col = f'{param}_volatility_std'
        n90_col = f'{param}_n90_days'
        
        if vol_col in df.columns and n90_col in df.columns:
            x = df[vol_col].dropna()
            y = df.loc[x.index, n90_col]
            
            if len(x) > 0:
                ax.scatter(x, y, alpha=0.6, s=60, c='steelblue', edgecolors='black', linewidth=0.5)
                
                # Add trend line
                if len(x) > 1:
                    try:
                        z = np.polyfit(x, y, 1)
                        p = np.poly1d(z)
                        x_trend = np.linspace(x.min(), x.max(), 100)
                        ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
                    except:
                        pass
            
        ax.set_xlabel('Volatility (σ)', fontsize=10)
        ax.set_ylabel('N_90 Days', fontsize=10)
        ax.set_title(f'{title}\nStability vs Speed', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
    # Hide empty subplots
    for i in range(len(params), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/volatility_vs_n90.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/volatility_vs_n90.png")
    plt.close()
    
    # ========================================================================
    # 3. Correlation Matrix (N_90 times)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    
    n90_cols = [f'{p}_n90_days' for p in params]
    valid_cols = [c for c in n90_cols if c in df.columns]
    
    if valid_cols:
        corr_data = df[valid_cols].dropna() 
        if len(corr_data) < 5:
            corr_matrix = df[valid_cols].corr() # Pairwise fallback
        else:
            corr_matrix = corr_data.corr()
            
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        short_labels = ['ZAMS', 'M_dot', '56Ni', 'Ek', 'Beta', 'T_exp', 'Av']
        valid_labels = [short_labels[i] for i, c in enumerate(n90_cols) if c in valid_cols]
        
        ax.set_xticks(range(len(valid_labels)))
        ax.set_yticks(range(len(valid_labels)))
        ax.set_xticklabels(valid_labels, rotation=45, ha='right')
        ax.set_yticklabels(valid_labels)
        
        for i in range(len(valid_labels)):
            for j in range(len(valid_labels)):
                val = corr_matrix.iloc[i, j]
                if not np.isnan(val):
                    text = ax.text(j, i, f'{val:.2f}',
                                  ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='Correlation Coefficient')
        ax.set_title('N_90 Convergence Time Correlations', fontsize=14, fontweight='bold', pad=20)
    else:
        ax.text(0.5, 0.5, 'Not enough data for correlations', ha='center', va='center')
    
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/n90_correlations.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/n90_correlations.png")
    plt.close()

    # ========================================================================
    # 3b. Parameter Value Correlation Matrix (Physical Degeneracies)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    
    final_cols = [f'{p}_final' for p in params]
    valid_cols = [c for c in final_cols if c in df.columns]
    
    if valid_cols:
        # Use simple correlation on final values
        corr_data = df[valid_cols].dropna() 
        if len(corr_data) < 5:
            corr_matrix = df[valid_cols].corr()
        else:
            corr_matrix = corr_data.corr()
            
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        short_labels = ['ZAMS', 'M_dot', '56Ni', 'Ek', 'Beta', 'T_exp', 'Av']
        valid_labels = [short_labels[i] for i, c in enumerate(final_cols) if c in valid_cols]
        
        ax.set_xticks(range(len(valid_labels)))
        ax.set_yticks(range(len(valid_labels)))
        ax.set_xticklabels(valid_labels, rotation=45, ha='right')
        ax.set_yticklabels(valid_labels)
        
        for i in range(len(valid_labels)):
            for j in range(len(valid_labels)):
                val = corr_matrix.iloc[i, j]
                if not np.isnan(val):
                    text = ax.text(j, i, f'{val:.2f}',
                                  ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='Correlation Coefficient')
        ax.set_title('Physical Parameter Correlations (Final Values)', fontsize=14, fontweight='bold', pad=20)
    else:
        ax.text(0.5, 0.5, 'Not enough data for parameter correlations', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/parameter_correlations.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/parameter_correlations.png")
    plt.close()



    
    # ========================================================================
    # 8. Overall Performance Summary
    # ========================================================================
    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
    
    short_labels = ['ZAMS', 'M_dot', '56Ni', 'Ek', 'Beta', 'T_exp', 'Av']
    
    # Convergence rates
    ax1 = fig.add_subplot(gs[0, :])
    convergence_rates = []
    for p in params:
        col = f'{p}_converged'
        rate = df[col].sum() / len(df) * 100 if col in df.columns else 0
        convergence_rates.append(rate)
        
    bars = ax1.bar(short_labels, convergence_rates, color='steelblue', 
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    ax1.set_ylabel('Convergence Rate (%)', fontsize=12)
    ax1.set_title('Parameter Convergence Success Rate (5+ obs)', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 105])
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, rate in zip(bars, convergence_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Average N_90 times
    ax2 = fig.add_subplot(gs[1, 0])
    avg_n90 = []
    for p in params:
        col = f'{p}_n90_days'
        val = df[col].mean() if col in df.columns else 0
        avg_n90.append(val)
        
    ax2.barh(short_labels, avg_n90, color='#2ecc71', 
             edgecolor='black', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('Average Days', fontsize=11)
    ax2.set_title('Average N_90 Convergence Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    for i, (label, val) in enumerate(zip(short_labels, avg_n90)):
        if not np.isnan(val):
            ax2.text(val + 0.3, i, f'{val:.1f}d', va='center', fontsize=9, fontweight='bold')
    
    # Volatility comparison
    ax3 = fig.add_subplot(gs[1, 1])
    avg_vol = []
    for p in params:
        col = f'{p}_volatility_std'
        val = df[col].mean() if col in df.columns else 0
        avg_vol.append(val)
        
    ax3.barh(short_labels, avg_vol, color='#e74c3c',
             edgecolor='black', linewidth=1.5, alpha=0.8)
    ax3.set_xlabel('Average σ', fontsize=11)
    ax3.set_title('Average Parameter Volatility', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Average Relative Uncertainty (NEW)
    ax5 = fig.add_subplot(gs[2, :])
    unc_cols = ['zams_rel_uncertainty', 'mloss_rate_rel_uncertainty', '56Ni_rel_uncertainty',
                'k_energy_rel_uncertainty', 'beta_rel_uncertainty', 'texp_rel_uncertainty', 
                'A_v_rel_uncertainty']
    avg_unc = []
    
    for col in unc_cols:
        if col in df.columns:
            val = df[col].mean() * 100  # Convert to percentage
            avg_unc.append(val)
        else:
            avg_unc.append(0)
    
    colors = ['#2ecc71' if u < 20 else '#f39c12' if u < 50 else '#e74c3c' for u in avg_unc]
    bars5 = ax5.bar(short_labels, avg_unc, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax5.axhline(y=20, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
    ax5.axhline(y=50, color='orange', linestyle='--', alpha=0.7, linewidth=1.5)
    ax5.set_ylabel('Average Relative Uncertainty (%)', fontsize=12)
    ax5.set_title('Average Parameter Uncertainty Across All Objects', fontsize=14, fontweight='bold')
    max_unc = max(avg_unc) if avg_unc else 0
    ax5.set_ylim([0, max_unc * 1.2 if max_unc > 0 else 100])
    ax5.grid(True, alpha=0.3, axis='y')
    
    for bar, unc in zip(bars5, avg_unc):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{unc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Prediction accuracy
    ax4 = fig.add_subplot(gs[3, :])
    if 'mag_arr_rmse' in df.columns:
        rmse_data = df['mag_arr_rmse'].dropna()
        if len(rmse_data) > 0:
            ax4.hist(rmse_data, bins=25, alpha=0.7, color='purple', edgecolor='black')
            ax4.axvline(rmse_data.mean(), color='red', linestyle='--',
                       label=f'Mean: {rmse_data.mean():.3f} mag', linewidth=2)
            ax4.axvline(rmse_data.median(), color='orange', linestyle='--',
                       label=f'Median: {rmse_data.median():.3f} mag', linewidth=2)
            ax4.legend()
            
    ax4.set_xlabel('RMSE (magnitudes)', fontsize=11)
    ax4.set_ylabel('Number of Objects', fontsize=11)
    ax4.set_title('Light Curve Prediction Accuracy (Early vs Final)', 
                  fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/overall_summary.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/overall_summary.png")
    plt.close()
    
    # ========================================================================
    # 9. Confidence Grade Distribution
    # ========================================================================
    if 'confidence_grade' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart of grades
        ax1 = axes[0]
        grade_counts = df['confidence_grade'].value_counts()
        grade_order = ['A', 'B', 'C', 'D']
        grade_colors = {'A': '#27ae60', 'B': '#3498db', 'C': '#f39c12', 'D': '#e74c3c'}
        
        sizes = [grade_counts.get(g, 0) for g in grade_order]
        colors = [grade_colors[g] for g in grade_order]
        labels = [f'Grade {g}\n({grade_counts.get(g, 0)})' for g in grade_order]
        
        # Only plot non-zero slices
        non_zero_mask = [s > 0 for s in sizes]
        sizes_nz = [s for s, m in zip(sizes, non_zero_mask) if m]
        colors_nz = [c for c, m in zip(colors, non_zero_mask) if m]
        labels_nz = [l for l, m in zip(labels, non_zero_mask) if m]
        
        if sizes_nz:
            wedges, texts, autotexts = ax1.pie(sizes_nz, colors=colors_nz, labels=labels_nz,
                                               autopct='%1.1f%%', startangle=90,
                                               textprops={'fontsize': 11})
            ax1.set_title('Confidence Grade Distribution', fontsize=14, fontweight='bold')
        
        # Histogram of confidence scores
        ax2 = axes[1]
        if 'confidence_score' in df.columns:
            scores = df['confidence_score'].dropna()
            ax2.hist(scores, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
            ax2.axvline(scores.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {scores.mean():.2f}')
            ax2.axvline(scores.median(), color='orange', linestyle='--', linewidth=2,
                       label=f'Median: {scores.median():.2f}')
            
            # Add grade threshold lines
            ax2.axvline(0.75, color='#27ae60', linestyle=':', linewidth=1.5, alpha=0.7)
            ax2.axvline(0.50, color='#3498db', linestyle=':', linewidth=1.5, alpha=0.7)
            ax2.axvline(0.25, color='#f39c12', linestyle=':', linewidth=1.5, alpha=0.7)
            ax2.text(0.76, ax2.get_ylim()[1]*0.9, 'A', color='#27ae60', fontweight='bold')
            ax2.text(0.51, ax2.get_ylim()[1]*0.9, 'B', color='#3498db', fontweight='bold')
            ax2.text(0.26, ax2.get_ylim()[1]*0.9, 'C', color='#f39c12', fontweight='bold')
            ax2.text(0.05, ax2.get_ylim()[1]*0.9, 'D', color='#e74c3c', fontweight='bold')
            
            ax2.set_xlabel('Confidence Score', fontsize=12)
            ax2.set_ylabel('Number of Objects', fontsize=12)
            ax2.set_title('Confidence Score Distribution', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confidence_grades.png', dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {output_dir}/confidence_grades.png")
        plt.close()
    
    # ========================================================================
    # 10. Relative Uncertainty by Parameter
    # ========================================================================
    rel_unc_cols = [f'{p}_rel_uncertainty' for p in params]
    if any(c in df.columns for c in rel_unc_cols):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        valid_cols = [c for c in rel_unc_cols if c in df.columns]
        valid_labels = [short_labels[i] for i, c in enumerate(rel_unc_cols) if c in valid_cols]
        
        # Collect data for box plot
        data = []
        for col in valid_cols:
            col_data = df[col].dropna()
            # Cap at 2 for visualization (represents >200% uncertainty)
            col_data = col_data.clip(upper=2)
            data.append(col_data)
        
        if data:
            positions = np.arange(len(valid_labels))
            bp = ax.boxplot(data, positions=positions, widths=0.6,
                           patch_artist=True, showmeans=True,
                           meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
            
            colors = plt.cm.Set2.colors
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)
            
            ax.axhline(0.1, color='green', linestyle='--', linewidth=1.5, alpha=0.7, 
                      label='10% uncertainty (excellent)')
            ax.axhline(0.5, color='orange', linestyle='--', linewidth=1.5, alpha=0.7,
                      label='50% uncertainty (poor)')
            ax.axhline(1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                      label='100% uncertainty (unconstrained)')
            
            ax.set_xticks(positions)
            ax.set_xticklabels(valid_labels, fontsize=11)
            ax.set_ylabel('Relative Uncertainty (σ/median)', fontsize=12)
            ax.set_title('Parameter Relative Uncertainties', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/relative_uncertainties.png', dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {output_dir}/relative_uncertainties.png")
        plt.close()
    
    # ========================================================================
    # 11. Confidence Components Breakdown
    # ========================================================================
    component_cols = ['avg_constraint_score', 'prior_posterior_contraction', 
                      'phase_coverage_score', 'fit_quality_score']
    if any(c in df.columns for c in component_cols):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        titles = ['Constraint Score', 'Prior-Posterior Contraction', 
                  'Phase Coverage', 'Fit Quality']
        colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
        
        for idx, (col, title, color) in enumerate(zip(component_cols, titles, colors)):
            ax = axes[idx]
            if col in df.columns:
                data = df[col].dropna()
                if len(data) > 0:
                    ax.hist(data, bins=20, alpha=0.7, color=color, edgecolor='black')
                    ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2,
                              label=f'Mean: {data.mean():.2f}')
                    ax.axvline(data.median(), color='orange', linestyle='--', linewidth=2,
                              label=f'Median: {data.median():.2f}')
                    
                    ax.set_xlabel(title, fontsize=11)
                    ax.set_ylabel('Count', fontsize=11)
                    ax.set_title(f'{title} Distribution', fontsize=12, fontweight='bold')
                    ax.set_xlim([0, 1])
                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confidence_components.png', dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {output_dir}/confidence_components.png")
        plt.close()
    
    # ========================================================================
    # 12. Confidence vs Data Quality
    # ========================================================================
    if 'confidence_score' in df.columns and 'num_observations' in df.columns:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Confidence vs num observations
        ax1 = axes[0]
        data = df[['confidence_score', 'num_observations']].dropna()
        if len(data) > 0:
            ax1.scatter(data['num_observations'], data['confidence_score'],
                       alpha=0.6, s=60, c='steelblue', edgecolors='black', linewidth=0.5)
            if len(data) > 2:
                try:
                    z = np.polyfit(data['num_observations'], data['confidence_score'], 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(data['num_observations'].min(), 
                                         data['num_observations'].max(), 100)
                    ax1.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
                except:
                    pass
            ax1.set_xlabel('Number of Observations', fontsize=11)
            ax1.set_ylabel('Confidence Score', fontsize=11)
            ax1.set_title('Confidence vs Observation Count', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
        
        # Confidence vs phase span
        ax2 = axes[1]
        if 'phase_span' in df.columns:
            data = df[['confidence_score', 'phase_span']].dropna()
            if len(data) > 0:
                ax2.scatter(data['phase_span'], data['confidence_score'],
                           alpha=0.6, s=60, c='#2ecc71', edgecolors='black', linewidth=0.5)
                ax2.set_xlabel('Phase Span (days)', fontsize=11)
                ax2.set_ylabel('Confidence Score', fontsize=11)
                ax2.set_title('Confidence vs Phase Coverage', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3)
        
        # Confidence vs posterior predictive spread
        ax3 = axes[2]
        if 'posterior_predictive_spread' in df.columns:
            data = df[['confidence_score', 'posterior_predictive_spread']].dropna()
            if len(data) > 0:
                ax3.scatter(data['posterior_predictive_spread'], data['confidence_score'],
                           alpha=0.6, s=60, c='#e74c3c', edgecolors='black', linewidth=0.5)
                ax3.set_xlabel('Posterior Predictive Spread (mag)', fontsize=11)
                ax3.set_ylabel('Confidence Score', fontsize=11)
                ax3.set_title('Confidence vs Prediction Uncertainty', fontsize=12, fontweight='bold')
                ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confidence_vs_data.png', dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {output_dir}/confidence_vs_data.png")
        plt.close()
    
    # ========================================================================
    # Parameter Correlation Scatter Grid (2×3)
    # ========================================================================
    # Use the already combined dataframe for plotting
    merged = df.copy()

    # Define the 6 scatter panels to match the reference figure layout
    # Row 1: ZAMS vs (56Ni, mloss_rate, k_energy)
    # Row 2: k_energy vs (texp, 56Ni, A_v)
    scatter_panels = [
        # (x_param, y_param, x_label, y_label)
        ('56Ni_final',       'zams_final',      r'$^{56}$Ni (M$_\odot$)',           r'ZAMS (M$_\odot$)'),
        ('mloss_rate_final', 'zams_final',      r'$-\log_{10}\dot{M}$ (M$_\odot$ yr$^{-1}$)', r'ZAMS (M$_\odot$)'),
        ('k_energy_final',   'zams_final',      r'$E_k$ ($10^{51}$ erg)',           r'ZAMS (M$_\odot$)'),
        ('texp_final',       'k_energy_final',  r'$t_{\mathrm{exp}}$ (day)',        r'$E_k$ ($10^{51}$ erg)'),
        ('56Ni_final',       'k_energy_final',  r'$^{56}$Ni (M$_\odot$)',           r'$E_k$ ($10^{51}$ erg)'),
        ('A_v_final',        'k_energy_final',  r'$A_V$ (mag)',                     r'$E_k$ ($10^{51}$ erg)'),
    ]

    # REFITT parameter bounds (dashed reference lines)
    param_bounds = {
        'zams_final':      (9.0, 17.0),
        'k_energy_final':  (0.1, 5.0),
        '56Ni_final':      (0.001, 0.3),
        'mloss_rate_final': (0.0, 5.0),
        'texp_final':      (0.0, None),
        'A_v_final':       (0.0, None),
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    marker_color = '#e74c71'   # Pink-red like the reference
    edge_color = '#8b0000'

    # Try to load outliers
    try:
        outliers_df = pd.read_csv('scatter_outliers.csv')
    except Exception:
        outliers_df = None

    for idx, (x_col, y_col, x_label, y_label) in enumerate(scatter_panels):
        row, col_idx = divmod(idx, 3)
        ax = axes[row][col_idx]

        if x_col in merged.columns and y_col in merged.columns:
            plot_data = merged[[x_col, y_col, 'object_id']].dropna()

            if len(plot_data) > 0:
                x_vals = plot_data[x_col].values
                y_vals = plot_data[y_col].values

                # Plot line of best fit to give visual reference to outliers
                if len(x_vals) > 1:
                    m, c = np.polyfit(x_vals, y_vals, 1)
                    x_trend = np.linspace(min(x_vals), max(x_vals), 100)
                    ax.plot(x_trend, m * x_trend + c, "c--", alpha=0.6, linewidth=1.5, zorder=2)

                x_param = x_col.replace('_final', '')
                y_param = y_col.replace('_final', '')
                x_err_col = f'{x_param}_rel_uncertainty'
                y_err_col = f'{y_param}_rel_uncertainty'

                # Are there outliers for this panel?
                panel_outliers = pd.DataFrame()
                if outliers_df is not None:
                    panel_outliers = outliers_df[(outliers_df['x_param_name'] == x_col) & (outliers_df['y_param_name'] == y_col)]
                
                outlier_oids = panel_outliers['object_id'].tolist() if not panel_outliers.empty else []

                # Plot regular points (non-outliers)
                mask_regular = ~plot_data['object_id'].isin(outlier_oids)
                if mask_regular.any():
                    reg_data = plot_data[mask_regular]
                    reg_idx = plot_data.index[mask_regular]
                    
                    reg_x_err = np.abs(reg_data[x_col].values) * np.nan_to_num(merged.loc[reg_idx, x_err_col].values, nan=0) if x_err_col in merged.columns else None
                    reg_y_err = np.abs(reg_data[y_col].values) * np.nan_to_num(merged.loc[reg_idx, y_err_col].values, nan=0) if y_err_col in merged.columns else None
                    
                    ax.errorbar(reg_data[x_col].values, reg_data[y_col].values,
                               xerr=reg_x_err, yerr=reg_y_err,
                               fmt='o', markersize=7,
                               color=marker_color, ecolor='gray',
                               elinewidth=0.7, capsize=0,
                               markeredgecolor=edge_color, markeredgewidth=0.5,
                               alpha=0.75, zorder=5)

                # Plot outliers
                if not panel_outliers.empty:
                    # Use a colormap to give each outlier a unique color
                    colors = plt.cm.tab10(np.linspace(0, 1, len(panel_outliers)))
                    
                    for i, (_, out_row) in enumerate(panel_outliers.iterrows()):
                        oid = out_row['object_id']
                        x_val = out_row['x_param_value']
                        y_val = out_row['y_param_value']
                        dir_val = out_row['direction']
                        
                        out_idx = plot_data[plot_data['object_id'] == oid].index
                        if not out_idx.empty:
                            x_e = np.abs(x_val) * np.nan_to_num(merged.loc[out_idx, x_err_col].values[0], nan=0) if x_err_col in merged.columns else None
                            y_e = np.abs(y_val) * np.nan_to_num(merged.loc[out_idx, y_err_col].values[0], nan=0) if y_err_col in merged.columns else None
                            
                            ax.errorbar([x_val], [y_val],
                                       xerr=[x_e] if x_e is not None else None, 
                                       yerr=[y_e] if y_e is not None else None,
                                       fmt='D', markersize=8,
                                       color=colors[i], ecolor='gray',
                                       elinewidth=1.0, capsize=0,
                                       markeredgecolor='black', markeredgewidth=1.0,
                                       alpha=1.0, zorder=10, label=f"{oid} ({dir_val})")

                    ax.legend(fontsize=8, loc='best')

                # Add parameter bounds as dashed lines
                for param_col, bounds in param_bounds.items():
                    if param_col == y_col:
                        for b in bounds:
                            if b is not None:
                                ax.axhline(y=b, color='gray', linestyle='--',
                                          linewidth=0.8, alpha=0.5, zorder=1)
                    if param_col == x_col:
                        for b in bounds:
                            if b is not None:
                                ax.axvline(x=b, color='gray', linestyle='--',
                                          linewidth=0.8, alpha=0.5, zorder=1)

        ax.set_xlabel(x_label, fontsize=13)
        ax.set_ylabel(y_label, fontsize=13)
        ax.tick_params(axis='both', labelsize=11)
        ax.grid(True, alpha=0.15, linewidth=0.5)

        # Minor ticks
        ax.minorticks_on()
        ax.tick_params(which='minor', length=3)

    plt.suptitle('Parameter Correlations — Final Inferred Values',
                fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/parameter_scatter_grid.png', dpi=200, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/parameter_scatter_grid.png")
    plt.close()

    print(f"\n✨ All summary plots created in: {output_dir}/")

if __name__ == "__main__":
    create_summary_plots('data/convergence_metrics.csv', 'data/summary_plots')

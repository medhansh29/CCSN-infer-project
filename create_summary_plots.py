#!/usr/bin/env python3
"""
Create summary visualizations from batch convergence analysis results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def create_summary_plots(metrics_file: str = 'convergence_metrics.csv',
                        output_dir: str = 'summary_plots'):
    """
    Create summary visualizations from convergence metrics.
    
    Args:
        metrics_file: Path to convergence metrics CSV
        output_dir: Directory to save plots
    """
    # Load data
    try:
        df = pd.read_csv(metrics_file)
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
    # 4. Lead Time Distributions (L_90)
    # ========================================================================
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f1c40f', '#e67e22', '#1abc9c']

    for idx, (param, title) in enumerate(zip(params, titles)):
        ax = axes[idx]
        l90_col = f'{param}_L90'
        
        if l90_col in df.columns:
            data = df[l90_col].dropna()
            
            if len(data) > 0:
                ax.hist(data, bins=15, alpha=0.7, color=colors[idx % len(colors)], edgecolor='black')
                ax.axvline(data.mean(), color='red', linestyle='--',
                          label=f'Mean: {data.mean():.1f} d', linewidth=2)
                ax.axvline(data.median(), color='orange', linestyle='--',
                          label=f'Median: {data.median():.1f} d', linewidth=2)
                
                # Percent early
                pct_col = f'{param}_percent_early'
                if pct_col in df.columns:
                    avg_early = df[pct_col].mean()
                    ax.text(0.95, 0.95, f'Avg {avg_early:.0f}% early',
                           transform=ax.transAxes, ha='right', va='top', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            else:
                 ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
        
        ax.set_xlabel('Prediction Lead Time (L_90 days)', fontsize=10)
        ax.set_ylabel('Number of Objects', fontsize=10)
        ax.set_title(f'{title}\nLead Time', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide empty
    for i in range(len(params), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/lead_time_distributions.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/lead_time_distributions.png")
    plt.close()

    # ========================================================================
    # 5. Degeneracy Breaking (t_break)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define pairs to show
    param_pairs = [
        ('zams', 'k_energy'), ('zams', 'mloss_rate'), ('mloss_rate', '56Ni'),
        ('beta', 'texp'), ('k_energy', 'texp')
    ]
    labels = [f'{p1} vs {p2}' for p1, p2 in param_pairs]
    
    t_breaks = []
    valid_labels = []
    
    for (p1, p2), label in zip(param_pairs, labels):
        col = f't_break_{p1}_{p2}'
        if col in df.columns:
            breaks = df[col].dropna()
            if len(breaks) > 0:
                t_breaks.append(breaks)
                valid_labels.append(label)
    
    if t_breaks:
        positions = np.arange(len(valid_labels))
        bp = ax.boxplot(t_breaks, positions=positions, widths=0.6,
                       patch_artist=True, showmeans=True,
                       meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
        
        # Color boxes
        for patch, color in zip(bp['boxes'], plt.cm.Set3.colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(valid_labels, rotation=15, ha='right', fontsize=10)
        ax.set_ylabel('t_break (days)', fontsize=12)
        ax.set_title('Parameter Degeneracy Breaking Times', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No Degeneracy Breaking Data', ha='center', va='center')
        
    plt.tight_layout()
    plt.savefig(f'{output_dir}/degeneracy_breaking.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/degeneracy_breaking.png")
    plt.close()

    # ========================================================================
    # 6. Phase Binned Residuals
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    phase_cols = ['rmse_shock_cooling', 'rmse_plateau', 'rmse_radioactive_tail']
    phase_names = ['Shock Cooling\n(<20d)', 'Plateau\n(20-100d)', 'Radioactive Tail\n(>100d)']
    colors_phase = ['wheat', 'lightblue', 'lightcoral']
    
    phase_data = []
    valid_names = []
    valid_colors = []
    
    for col, name, color in zip(phase_cols, phase_names, colors_phase):
        if col in df.columns:
            data = df[col].dropna()
            if len(data) > 0:
                phase_data.append(data)
                valid_names.append(name)
                valid_colors.append(color)
    
    if phase_data:
        positions = np.arange(len(valid_names))
        bp = ax.boxplot(phase_data, positions=positions, widths=0.6,
                       patch_artist=True, showmeans=True,
                       meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
        
        for patch, color in zip(bp['boxes'], valid_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            
        ax.set_xticks(positions)
        ax.set_xticklabels(valid_names, fontsize=11)
        ax.set_ylabel('RMSE (magnitudes)', fontsize=12)
        ax.set_title('Prediction Error by Phase', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    else:
         ax.text(0.5, 0.5, 'No Phase RMSE Data', ha='center', va='center')
         
    plt.tight_layout()
    plt.savefig(f'{output_dir}/phase_binned_errors.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/phase_binned_errors.png")
    plt.close()

    # ========================================================================
    # 7. Lead Time vs Convergence (Scatter)
    # ========================================================================
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (param, title) in enumerate(zip(params, titles)):
        ax = axes[idx]
        n90_col = f'{param}_n90_days'
        l90_col = f'{param}_L90'
        
        if n90_col in df.columns and l90_col in df.columns:
            data = df[[n90_col, l90_col]].dropna()
            
            if len(data) > 0:
                ax.scatter(data[n90_col], data[l90_col], 
                          alpha=0.6, s=60, edgecolors='black', linewidth=0.5, c='purple')
                
                if len(data) > 1:
                    try:
                        z = np.polyfit(data[n90_col], data[l90_col], 1)
                        p = np.poly1d(z)
                        x_trend = np.linspace(data[n90_col].min(), data[n90_col].max(), 100)
                        ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
                    except:
                        pass
        
        ax.set_xlabel('Convergence Time (N_90)', fontsize=10)
        ax.set_ylabel('Lead Time (L_90)', fontsize=10)
        ax.set_title(f'{title}\nSpeed vs Lead', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide empty
    for i in range(len(params), len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(f'{output_dir}/lead_time_vs_convergence.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/lead_time_vs_convergence.png")
    plt.close()
    
    # ========================================================================
    # 8. Overall Performance Summary
    # ========================================================================
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
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
    
    # Prediction accuracy
    ax4 = fig.add_subplot(gs[2, :])
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
    
    plt.savefig(f'{output_dir}/overall_summary.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/overall_summary.png")
    plt.close()
    
    print(f"\n✨ All summary plots created in: {output_dir}/")

if __name__ == "__main__":
    create_summary_plots()

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
    df = pd.read_csv(metrics_file)
    Path(output_dir).mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. N_90 Distribution Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    params = ['zams', 'mloss_rate', '56Ni']
    titles = ['ZAMS (Progenitor Mass)', 'Mass-Loss Rate', '56Ni Mass']
    
    for ax, param, title in zip(axes, params, titles):
        data = df[f'{param}_n90_days'].dropna()
        
        ax.hist(data, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(data.mean(), color='red', linestyle='--', 
                   label=f'Mean: {data.mean():.1f} days', linewidth=2)
        ax.axvline(data.median(), color='orange', linestyle='--',
                   label=f'Median: {data.median():.1f} days', linewidth=2)
        
        ax.set_xlabel('Days to 10% Convergence (N_90)', fontsize=11)
        ax.set_ylabel('Number of Objects', fontsize=11)
        ax.set_title(f'{title}\nConvergence Time Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/n90_distributions.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/n90_distributions.png")
    plt.close()
    
    # 2. Volatility vs N_90 Scatter
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, param, title in zip(axes, params, titles):
        x = df[f'{param}_volatility_std'].dropna()
        y = df.loc[x.index, f'{param}_n90_days']
        
        ax.scatter(x, y, alpha=0.6, s=60, c='steelblue', edgecolors='black', linewidth=0.5)
        
        # Add trend line
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Volatility (σ)', fontsize=11)
        ax.set_ylabel('N_90 Days', fontsize=11)
        ax.set_title(f'{title}\nStability vs Convergence Speed', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/volatility_vs_n90.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/volatility_vs_n90.png")
    plt.close()
    
    # 3. Correlation Matrix (N_90 times)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    n90_cols = ['zams_n90_days', 'mloss_rate_n90_days', '56Ni_n90_days']
    corr_data = df[n90_cols].dropna()
    corr_matrix = corr_data.corr()
    
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Add labels
    labels = ['ZAMS', 'Mass-Loss Rate', '56Ni']
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    
    # Add correlation values
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Correlation Coefficient')
    ax.set_title('N_90 Convergence Time Correlations', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/n90_correlations.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/n90_correlations.png")
    plt.close()
    
    # 4. Overall Performance Summary
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Convergence rates
    ax1 = fig.add_subplot(gs[0, :])
    convergence_rates = [df[f'{p}_converged'].sum() / len(df) * 100 for p in params]
    bars = ax1.bar(labels, convergence_rates, color=['#2ecc71', '#3498db', '#e74c3c'], 
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    ax1.set_ylabel('Convergence Rate (%)', fontsize=12)
    ax1.set_title('Parameter Convergence Success Rate (10+ observations)', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 105])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for bar, rate in zip(bars, convergence_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Average N_90 times
    ax2 = fig.add_subplot(gs[1, 0])
    avg_n90 = [df[f'{p}_n90_days'].mean() for p in params]
    ax2.barh(labels, avg_n90, color=['#2ecc71', '#3498db', '#e74c3c'], 
             edgecolor='black', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('Average Days', fontsize=11)
    ax2.set_title('Average N_90 Convergence Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (label, val) in enumerate(zip(labels, avg_n90)):
        ax2.text(val + 0.3, i, f'{val:.1f}d', va='center', fontsize=10, fontweight='bold')
    
    # Volatility comparison
    ax3 = fig.add_subplot(gs[1, 1])
    avg_vol = [df[f'{p}_volatility_std'].mean() for p in params]
    ax3.barh(labels, avg_vol, color=['#2ecc71', '#3498db', '#e74c3c'],
             edgecolor='black', linewidth=1.5, alpha=0.8)
    ax3.set_xlabel('Average σ', fontsize=11)
    ax3.set_title('Average Parameter Volatility', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Prediction accuracy
    ax4 = fig.add_subplot(gs[2, :])
    rmse_data = df['mag_arr_rmse'].dropna()
    ax4.hist(rmse_data, bins=25, alpha=0.7, color='purple', edgecolor='black')
    ax4.axvline(rmse_data.mean(), color='red', linestyle='--',
               label=f'Mean: {rmse_data.mean():.3f} mag', linewidth=2)
    ax4.axvline(rmse_data.median(), color='orange', linestyle='--',
               label=f'Median: {rmse_data.median():.3f} mag', linewidth=2)
    ax4.set_xlabel('RMSE (magnitudes)', fontsize=11)
    ax4.set_ylabel('Number of Objects', fontsize=11)
    ax4.set_title('Light Curve Prediction Accuracy (Early vs Final)', 
                  fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.savefig(f'{output_dir}/overall_summary.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_dir}/overall_summary.png")
    plt.close()
    
    print(f"\n✨ All summary plots created in: {output_dir}/")


if __name__ == "__main__":
    create_summary_plots()

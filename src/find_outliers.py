import pandas as pd
import numpy as np
import argparse

def find_all_outliers(metrics_file='data/convergence_metrics.csv'):
    df = pd.read_csv(metrics_file)
    
    outliers = []
    
    scatter_panels = [
        ('56Ni_final', 'zams_final', 'ZAMS-Ni'),
        ('mloss_rate_final', 'zams_final', 'ZAMS-Mloss'),
        ('k_energy_final', 'zams_final', 'ZAMS-Ek'),
        ('texp_final', 'k_energy_final', 'Ek-Texp'),
        ('56Ni_final', 'k_energy_final', 'Ek-Ni'),
        ('A_v_final', 'k_energy_final', 'Ek-Av'),
    ]
    
    for x_col, y_col, type_name in scatter_panels:
        valid_df = df.dropna(subset=[x_col, y_col])
        if len(valid_df) > 1:
            x = valid_df[x_col].values
            y = valid_df[y_col].values
            
            # Line of best fit
            m, c = np.polyfit(x, y, 1)
            
            # Expected y
            y_expected = m * x + c
            residuals = y - y_expected
            
            # Orthogonal distance
            distances = np.abs(m * x - y + c) / np.sqrt(m**2 + 1)
            
            valid_df = valid_df.copy()
            valid_df['distance_from_trend'] = distances
            valid_df['residual'] = residuals
            
            # Direction
            valid_df['direction'] = np.where(residuals > 0, 'Above Trendline', 'Below Trendline')
            
            # Top 3 or 4 outliers
            num_outliers = 4 if 'Ni' in type_name else 3
            
            top_outliers = valid_df.sort_values(by='distance_from_trend', ascending=False).head(num_outliers)
            
            for _, row in top_outliers.iterrows():
                outliers.append({
                    'object_id': row['object_id'],
                    'outlier_type': type_name,
                    'x_param_name': x_col,
                    'y_param_name': y_col,
                    'x_param_value': row[x_col],
                    'y_param_value': row[y_col],
                    'distance_from_trend': row['distance_from_trend'],
                    'direction': row['direction'],
                    'residual': row['residual']
                })
                
    outliers_df = pd.DataFrame(outliers)
    outliers_df.to_csv('data/scatter_outliers.csv', index=False)
    print("Saved outliers to data/scatter_outliers.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Identify outliers in CC SN parameter spaces.")
    parser.add_argument('--metrics_file', type=str, default='data/convergence_metrics.csv',
                        help='Input metrics CSV file')
    args = parser.parse_args()
    find_all_outliers(metrics_file=args.metrics_file)

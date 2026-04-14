import pandas as pd
import numpy as np
from scipy.stats import chi2
import os

class BivariateOutlierDetector:
    """Helper class to find 2D outlier points based on linear trendlines."""
    
    @staticmethod
    def detect(df: pd.DataFrame, output_path: str = "data/scatter_outliers.csv") -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
            
        combinations = [
            {'name': 'Mloss-Ek', 'x': 'mloss_rate_final', 'y': 'k_energy_final'},
            {'name': 'Ek-Ni', 'x': 'k_energy_final', 'y': '56Ni_final'},
            {'name': 'Texp-Beta', 'x': 'texp_final', 'y': 'beta_final'},
            {'name': 'logZ-Av', 'x': 'logZ_final', 'y': 'A_v_final'}
        ]
        
        all_outliers = []
        for combo in combinations:
            x_col, y_col = combo['x'], combo['y']
            if x_col not in df.columns or y_col not in df.columns:
                continue
                
            plot_data = df[['object_id', x_col, y_col]].dropna()
            if len(plot_data) < 3:
                continue
                
            X = plot_data[x_col].values
            Y = plot_data[y_col].values
            
            m, c = np.polyfit(X, Y, 1)
            Y_pred = m * X + c
            residuals = Y - Y_pred
            std_res = np.std(residuals)
            if std_res == 0:
                continue
                
            plot_data['distance_from_trend'] = np.abs(residuals) / std_res
            plot_data['residual'] = residuals
            plot_data['direction'] = np.where(residuals > 0, "Above Trendline", "Below Trendline")
            
            # Keep top 5 deviations
            top_outliers = plot_data.nlargest(5, 'distance_from_trend')
            
            for _, row in top_outliers.iterrows():
                all_outliers.append({
                    'object_id': row['object_id'],
                    'outlier_type': combo['name'],
                    'x_param_name': x_col,
                    'y_param_name': y_col,
                    'x_param_value': row[x_col],
                    'y_param_value': row[y_col],
                    'distance_from_trend': row['distance_from_trend'],
                    'direction': row['direction'],
                    'residual': row['residual']
                })
                
        out_df = pd.DataFrame(all_outliers)
        
        # Overwrite the existing CSV with base 2D outliers
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not out_df.empty:
            out_df.to_csv(output_path, index=False)
            print(f"✅ Created {output_path} with 2D bivariate outliers")
        else:
            # Create empty template
            pd.DataFrame(columns=['object_id', 'outlier_type', 'x_param_name', 'y_param_name', 
                                  'x_param_value', 'y_param_value', 'distance_from_trend', 
                                  'direction', 'residual']).to_csv(output_path, index=False)
        return out_df

class MultivariateOutlierDetector:
    """Helper class to find multi-dimensional outliers based on Mahalanobis distance."""
    
    @staticmethod
    def detect(df: pd.DataFrame, output_path: str = "data/scatter_outliers.csv") -> pd.DataFrame:
        """
        Calculates multivariate physical clusters using Mahalanobis distance to flag anomalies.
        Appends the new 3D outlier records to the existing scatter_outliers.csv file if it exists.
        
        Args:
            df: The main convergence dataframe containing *_final values
            output_path: Path to the outliers CSV file
            
        Returns:
            DataFrame of just the newly detected multivariate outliers
        """
        if df.empty:
            return pd.DataFrame()
            
        clusters = {
            "Energy Engine": ['k_energy_final', 'mloss_rate_final', '56Ni_final'],
            "Progenitor Evolution": ['zams_final', 'mloss_rate_final', 'logZ_final'],
            "Modeling Degeneracy": ['A_v_final', 'texp_final', 'logZ_final'],
            "Ejecta Efficiency": ['k_energy_final', 'mloss_rate_final', 'beta_final'],
            "LC Morphology": ['texp_final', '56Ni_final', 'A_v_final']
        }
        
        new_outliers = []
        threshold_dist = chi2.ppf(0.99, df=3)
        
        for cluster_name, params in clusters.items():
            missing = [p for p in params if p not in df.columns]
            if missing:
                print(f"Skipping {cluster_name} cluster detection due to missing params: {missing}")
                continue
                
            valid_df = df.dropna(subset=params).copy()
            if len(valid_df) < 5:
                continue
                
            x = valid_df[params].values
            center = np.mean(x, axis=0)
            try:
                cov = np.cov(x, rowvar=False)
                inv_cov = np.linalg.pinv(cov)
                
                diff = x - center
                m_distances = []
                for idx in range(len(valid_df)):
                    d_squared = np.dot(np.dot(diff[idx], inv_cov), diff[idx].T)
                    m_distances.append(d_squared)
                
                valid_df['mahalanobis_dist'] = m_distances
                outlier_rows = valid_df[valid_df['mahalanobis_dist'] > threshold_dist]
                
                for _, row in outlier_rows.iterrows():
                    new_outliers.append({
                        'object_id': row['object_id'],
                        'outlier_type': cluster_name,
                        'x_param_name': params[0],
                        'y_param_name': params[1],
                        'x_param_value': row[params[2]],
                        'y_param_value': row[params[1]],
                        'distance_from_trend': row['mahalanobis_dist'],
                        'direction': "Above Trendline" if row['mahalanobis_dist'] > threshold_dist else "Below Trendline",
                        'residual': row['mahalanobis_dist'] - threshold_dist
                    })
            except Exception as e:
                print(f"Error computing Mahalanobis dist for {cluster_name}: {e}")
                
        new_outliers_df = pd.DataFrame(new_outliers)
        
        if os.path.exists(output_path):
            existing_df = pd.read_csv(output_path)
            if not new_outliers_df.empty:
                existing_df = existing_df[~existing_df['outlier_type'].isin(clusters.keys())]
                final_df = pd.concat([existing_df, new_outliers_df], ignore_index=True)
                final_df.to_csv(output_path, index=False)
                print(f"✅ Appended multivariate outliers to {output_path}")
        else:
            if not new_outliers_df.empty:
                new_outliers_df.to_csv(output_path, index=False)
                print(f"✅ Created {output_path} with multivariate outliers")
                
        return new_outliers_df

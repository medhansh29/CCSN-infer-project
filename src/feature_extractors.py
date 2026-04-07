import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from scipy.stats import linregress, gaussian_kde
from scipy.signal import find_peaks
import os

try:
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u
    COSMO = FlatLambdaCDM(H0=70, Om0=0.3)
except ImportError:
    COSMO = None

class GPMorphologicalExtractor:
    @staticmethod
    def extract(raw_df: pd.DataFrame, t_exp_mjd: float = None):
        """
        Step 1: GP Morphological Extraction.
        Extracts M_plateau_25d (at t_exp + 25d) rather than M_peak to avoid shock
        breakout spike contamination in the first 0-10 days.
        Also computes t_fall, t_rise, and g-r color slope.
        """
        result = {'t_fall': None, 'M_plateau_25d': None, 't_rise': None, 'gr_slope': None}
        if raw_df is None or len(raw_df) < 5:
            return result

        band_counts = raw_df['filter'].value_counts()
        if band_counts.empty: return result
        dominant_band = band_counts.idxmax()

        def fit_band(df_b):
            if len(df_b) < 5: return None, None, None
            X = df_b['mjd'].values[:, np.newaxis]
            y = df_b['mag'].values
            dy = df_b['magerr'].values
            kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-5, 1e1))
            gp = GaussianProcessRegressor(kernel=kernel, alpha=dy**2, n_restarts_optimizer=5, normalize_y=True)
            try:
                gp.fit(X, y)
                return gp, X.min(), X.max()
            except Exception:
                return None, None, None

        df_dom = raw_df[raw_df['filter'] == dominant_band].sort_values('mjd')
        gp_dom, x_min, x_max = fit_band(df_dom)

        if gp_dom is not None:
            X_pred = np.linspace(x_min, x_max, 500)[:, np.newaxis]
            y_pred, _ = gp_dom.predict(X_pred, return_std=True)

            # t_fall: index of largest positive magnitude derivative (fastest decline)
            dy_dx = np.gradient(y_pred, X_pred[:, 0])
            t_fall_idx = np.argmax(dy_dx)
            result['t_fall'] = X_pred[t_fall_idx, 0]

            # --- M_plateau_25d (Phase 3 fix: avoid shock breakout spike) ---
            # Use t_exp_mjd if provided (from REFITT texp), otherwise fall back to
            # the first >3sigma detection as a proxy.
            plateau_eval_mjd = None
            if t_exp_mjd is not None and not np.isnan(t_exp_mjd):
                plateau_eval_mjd = t_exp_mjd + 25.0
            else:
                df_dom_snr = df_dom.copy()
                df_dom_snr['snr'] = 1.0857 / df_dom_snr['magerr']
                high_snr = df_dom_snr[df_dom_snr['snr'] > 3]
                if not high_snr.empty:
                    first_3sig = high_snr['mjd'].min()
                    plateau_eval_mjd = first_3sig + 25.0

            if plateau_eval_mjd is not None:
                if x_min <= plateau_eval_mjd <= x_max:
                    y_plat, _ = gp_dom.predict([[plateau_eval_mjd]], return_std=True)
                    result['M_plateau_25d'] = float(y_plat[0])
                else:
                    # Fallback: find the brightest magnitude in the GP fit within observed range
                    result['M_plateau_25d'] = float(np.min(y_pred))  # Min mag is max luminosity

            # Rise time: t_exp to earliest high-SNR detection
            df_dom_snr = df_dom.copy()
            df_dom_snr['snr'] = 1.0857 / df_dom_snr['magerr']
            high_snr = df_dom_snr[df_dom_snr['snr'] > 3]
            if not high_snr.empty and plateau_eval_mjd is not None:
                first_3sig = high_snr['mjd'].min()
                plateau_mjd = plateau_eval_mjd - 25.0  # back to t_exp proxy
                if first_3sig > (plateau_mjd - 30):  # sanity window
                    result['t_rise'] = plateau_eval_mjd - 25.0 - first_3sig

            # Color Evolution: plateau days 20-60 post explosion
            g_bands = [b for b in band_counts.keys() if 'g' in str(b).lower()]
            r_bands = [b for b in band_counts.keys() if 'r' in str(b).lower()]
            if g_bands and r_bands and plateau_eval_mjd is not None:
                df_g = raw_df[raw_df['filter'] == g_bands[0]].sort_values('mjd')
                df_r = raw_df[raw_df['filter'] == r_bands[0]].sort_values('mjd')
                gp_g, g_min, g_max = fit_band(df_g)
                gp_r, r_min, r_max = fit_band(df_r)
                if gp_g is not None and gp_r is not None:
                    t_plat_start = plateau_eval_mjd  # t_exp + 25
                    t_plat_end = plateau_eval_mjd + 35  # up to t_exp + 60
                    if max(g_min, r_min) <= t_plat_start and min(g_max, r_max) >= t_plat_end:
                        eval_mz = np.linspace(t_plat_start, t_plat_end, 40)[:, np.newaxis]
                        g_pred = gp_g.predict(eval_mz)
                        r_pred = gp_r.predict(eval_mz)
                        color_gr = g_pred - r_pred
                        phases = eval_mz[:, 0] - (plateau_eval_mjd - 25.0)
                        slope, _, _, _, _ = linregress(phases, color_gr)
                        result['gr_slope'] = slope

        return result

class MassBudgetExtractor:
    @staticmethod
    def extract(final_params: dict):
        """
        Step 2b: Mass Budget Sanity Check
        Proxy checks if Mej is enough to sustain plateau dynamics using ZAMS mass scaling.
        """
        result = {'implied_Mej': None, 'mass_budget_violation': False}
        if 'zams' not in final_params or 'k_energy' not in final_params: 
            return result
            
        get_val = lambda x: final_params[x][0] if isinstance(final_params[x], list) else float(final_params[x])
        try:
            zams = get_val('zams')
            k_energy = get_val('k_energy')
            
            implied_mej = zams - 1.5
            result['implied_Mej'] = implied_mej
            
            # Massive envelope stripping check
            if k_energy / implied_mej > 1.0:
                result['mass_budget_violation'] = True
        except Exception:
            pass
            
        return result

class LateTimeTailRegression:
    @staticmethod
    def extract(raw_df: pd.DataFrame, peak_mjd: float):
        """
        Step 2: Late-Time Tail Regression
        Perform linear regression on data Phase > 120 days.
        """
        result = {'tail_slope': None, 'late_time_r2': None}
        if raw_df is None or peak_mjd is None:
            return result
            
        # Calculate Phase = mjd - peak_mjd
        raw_df = raw_df.copy()
        raw_df['phase'] = raw_df['mjd'] - peak_mjd
        late_df = raw_df[raw_df['phase'] > 120].sort_values('phase')
        
        if len(late_df) < 3:
            return result
            
        slope, intercept, r_value, p_value, std_err = linregress(late_df['phase'], late_df['mag'])
        
        result['tail_slope'] = slope
        result['late_time_r2'] = r_value**2
        return result

class SystematicResidualsAnalyzer:
    @staticmethod
    def extract(raw_df: pd.DataFrame, model_df: pd.DataFrame):
        """
        Step 3: Systematic Residuals Analysis
        Calculate autocorrelation of residuals.
        """
        result = {'lag1_autocorr': None, 'residual_std': None}
        if raw_df is None or model_df is None or len(raw_df) == 0 or len(model_df) == 0:
            return result
            
        # We need to map raw MJD to model MJD correctly. 
        # Interpolate model_df (which is smooth) to raw_df MJDs
        residuals = []
        for _, row in raw_df.iterrows():
            mjd = row['mjd']
            mag = row['mag']
            closest_idx = (np.abs(model_df['mjd_arr'] - mjd)).idxmin() if 'mjd_arr' in model_df else None
            if closest_idx is not None:
                # Assuming model mag is median
                model_mag = model_df.loc[closest_idx, 'mag_median']
                residuals.append(mag - model_mag)
                
        if len(residuals) < 5:
            return result
            
        residuals = np.array(residuals)
        if np.std(residuals) > 1e-9:
            lag1 = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        else:
            lag1 = 0
            
        result['lag1_autocorr'] = lag1
        result['residual_std'] = np.std(residuals)
        
        return result

class PrecursorScan:
    ZTF_LIMITING_MAG = 20.5
    RSG_ERUPT_ABS_MAG = -13.0

    @staticmethod
    def extract(raw_df: pd.DataFrame, t_exp_mjd: float, distance_modulus: float = None):
        result = {
            'precursor_flag': False,
            'precursor_status': 'not_scanned'
        }
        if raw_df is None or t_exp_mjd is None or np.isnan(t_exp_mjd):
            return result

        if distance_modulus is not None and not np.isnan(distance_modulus):
            expected_app_mag = PrecursorScan.RSG_ERUPT_ABS_MAG + distance_modulus
            if expected_app_mag > PrecursorScan.ZTF_LIMITING_MAG:
                result['precursor_status'] = 'theoretically_undetectable'
                return result

        prepeak_df = raw_df[(raw_df['mjd'] >= t_exp_mjd - 10) & (raw_df['mjd'] < t_exp_mjd)].copy()
        baseline_df = raw_df[raw_df['mjd'] < t_exp_mjd - 10].copy()

        if len(prepeak_df) < 2 or len(baseline_df) < 3:
            result['precursor_status'] = 'insufficient_coverage'
            return result

        baseline_df['flux'] = 10.0 ** (-0.4 * baseline_df['mag'])
        mu = baseline_df['flux'].mean()
        sigma = baseline_df['flux'].std()
        if pd.isna(sigma) or sigma == 0:
            sigma = baseline_df['flux_err'].mean() if 'flux_err' in baseline_df else mu * 0.1

        prepeak_df = prepeak_df.sort_values('mjd')
        prepeak_df['flux'] = 10.0 ** (-0.4 * prepeak_df['mag'])
        
        consecutive_count = 0
        threshold = mu + 3 * sigma
        for flux in prepeak_df['flux']:
            if flux > threshold:
                consecutive_count += 1
                if consecutive_count >= 2:
                    result['precursor_flag'] = True
                    result['precursor_status'] = 'detected'
                    break
            else:
                consecutive_count = 0
        
        if not result['precursor_flag']:
            result['precursor_status'] = 'not_detected'

        return result

class EarlyRiseExcessExtractor:
    @staticmethod
    def extract(raw_df: pd.DataFrame, t_exp_mjd: float):
        result = {'early_rise_excess_flag': False}
        if raw_df is None or t_exp_mjd is None or np.isnan(t_exp_mjd):
            return result
        
        early_df = raw_df[(raw_df['mjd'] > t_exp_mjd) & (raw_df['mjd'] <= t_exp_mjd + 3)].copy()
        if len(early_df) < 3:
            return result
        
        early_df = early_df.sort_values('mjd')
        early_df['dt'] = early_df['mjd'] - t_exp_mjd
        early_df['flux'] = 10.0 ** (-0.4 * early_df['mag'])
        
        from scipy.optimize import curve_fit
        def fireball(t, a):
            return a * (t ** 2)
            
        try:
            popt, _ = curve_fit(fireball, early_df['dt'], early_df['flux'])
            expected_flux = fireball(early_df['dt'].values, *popt)
            expected_mag = -2.5 * np.log10(expected_flux)
            
            diffs = expected_mag - early_df['mag'].values
            if np.any(diffs > 0.1):
                result['early_rise_excess_flag'] = True
        except:
            pass
            
        return result

class ArrestedCoolingExtractor:
    @staticmethod
    def extract(raw_df: pd.DataFrame, t_exp_mjd: float):
        result = {'arrested_cooling_flag': False, 'early_gr_slope': None}
        if raw_df is None or t_exp_mjd is None or np.isnan(t_exp_mjd):
            return result
            
        band_counts = raw_df['filter'].value_counts()
        g_bands = [b for b in band_counts.keys() if 'g' in str(b).lower()]
        r_bands = [b for b in band_counts.keys() if 'r' in str(b).lower()]
        
        if not g_bands or not r_bands:
            return result
            
        early_df = raw_df[(raw_df['mjd'] > t_exp_mjd) & (raw_df['mjd'] <= t_exp_mjd + 15)].copy()
        
        g_early = early_df[early_df['filter'] == g_bands[0]]
        r_early = early_df[early_df['filter'] == r_bands[0]]
        
        if len(g_early) < 2 or len(r_early) < 2:
            return result
            
        try:
            r_interp = np.interp(g_early['mjd'].values, r_early['mjd'].values, r_early['mag'].values)
            g_minus_r = g_early['mag'].values - r_interp
            
            slope, _, _, _, _ = linregress(g_early['mjd'].values, g_minus_r)
            result['early_gr_slope'] = slope
            
            if slope < 0.04:
                result['arrested_cooling_flag'] = True
        except:
            pass
            
        return result

class PlateauTopographyExtractor:
    @staticmethod
    def extract(raw_df: pd.DataFrame, t_exp_mjd: float):
        result = {'rebrightening_flag': False, 'linear_residual_flag': False}
        if raw_df is None or t_exp_mjd is None or np.isnan(t_exp_mjd):
            return result
            
        plat_df = raw_df[(raw_df['mjd'] >= t_exp_mjd + 20) & (raw_df['mjd'] <= t_exp_mjd + 70)].copy()
        band_counts = plat_df['filter'].value_counts()
        if band_counts.empty:
            return result
            
        dom = band_counts.idxmax()
        df_dom = plat_df[plat_df['filter'] == dom].sort_values('mjd')
        
        if len(df_dom) < 5:
            return result
        
        mjd = df_dom['mjd'].values
        mag = df_dom['mag'].values
        
        dt = np.diff(mjd)
        dmag = np.diff(mag)
        slope = dmag / dt
        
        in_neg = False
        span_start = 0
        for i, s in enumerate(slope):
            if s < 0:
                if not in_neg:
                    in_neg = True
                    span_start = mjd[i]
            else:
                if in_neg:
                    in_neg = False
                    if mjd[i] - span_start > 5:
                        result['rebrightening_flag'] = True
                        break
        if in_neg and (mjd[-1] - span_start > 5):
            result['rebrightening_flag'] = True

        try:
            line_slope, line_int, _, _, _ = linregress(mjd, mag)
            expected_mag = line_slope * mjd + line_int
            residuals = expected_mag - mag
            
            cluster_count = 0
            for res in residuals:
                if res >= 0.1:
                    cluster_count += 1
                    if cluster_count >= 2:
                        result['linear_residual_flag'] = True
                        break
                else:
                    cluster_count = 0
        except:
            pass
            
        return result

class PosteriorAnalyzer:
    @staticmethod
    def extract(samples_path: str, parameters: dict):
        """
        Step 5b: Bimodal Posterior Warning & Prior Bounds Pegging
        Reads raw MCMC samples and checks if the median is unphysical or bounded.
        """
        result = {'is_bimodal': False, 'prior_pegged': []}
        
        if not os.path.exists(samples_path): 
            return result
            
        try:
            samples = np.loadtxt(samples_path)
        except Exception:
            return result
            
        num_params = min(samples.shape[1], len(parameters))
        param_keys = list(parameters.keys())
        
        for i in range(num_params):
            col_samples = samples[:, i]
            param_name = param_keys[i]
            
            s_min, s_max = np.min(col_samples), np.max(col_samples)
            s_range = s_max - s_min
            
            # The JSON parameters are [median, lower_err, upper_err]
            if isinstance(parameters[param_name], list):
                median = parameters[param_name][0]
            else:
                median = float(parameters[param_name])
            
            if s_range > 1e-5:
                # Prior boundary pegging: if median is within 5% of the absolute MCMC sample extremes
                if (median - s_min) / s_range < 0.05 or (s_max - median) / s_range < 0.05:
                    result['prior_pegged'].append(param_name)
                    
                # Bimodality check via Gaussian KDE
                kde = gaussian_kde(col_samples)
                x_eval = np.linspace(s_min, s_max, 100)
                y_eval = kde(x_eval)
                
                # Find peaks with prominence > 10% of maximum density
                peaks, _ = find_peaks(y_eval, prominence=np.max(y_eval)*0.1)
                
                # If there are two or more distinct peaks, we flag it.
                if len(peaks) >= 2:
                    result['is_bimodal'] = True
                    
        return result

class PriorsVolatilityCheck:
    @staticmethod
    def extract(final_params: dict, prior_dict: dict = None, previous_runs: list = None):
        """
        Step 5: Priors & Volatility Check
        Calculate deviations from prior mean and rolling variance.
        """
        if prior_dict is None:
            # Default generic priors if none passed
            prior_dict = {
                'zams': (15.0, 4.0), 'mloss_rate': (3.0, 1.5), '56Ni': (0.05, 0.03),
                'k_energy': (1.0, 0.5), 'beta': (3.0, 1.0), 'texp': (10.0, 5.0), 'A_v': (10.0, 5.0)
            }
            
        result = {'prior_deviations': {}, 'volatility_scores': {}}
        
        for param, values in final_params.items():
            median_val = values[0] if isinstance(values, list) else values
            if param in prior_dict:
                p_mean, p_std = prior_dict[param]
                z_score = abs(median_val - p_mean) / (p_std + 1e-9)
                result['prior_deviations'][param] = z_score
                
        if previous_runs and len(previous_runs) > 1:
            for param in final_params.keys():
                history = []
                for run in previous_runs:
                    if param in run:
                        val = run[param]
                        if isinstance(val, (list, tuple)):
                            history.append(val[0])
                        else:
                            history.append(val)
                if len(history) > 1:
                    result['volatility_scores'][param] = np.var(history)
                    
        return result

class Aggregator:
    # The morphological/physical feature space for the Isolation Forest.
    # Deliberately excludes REFITT model parameters (zams, beta, k_energy, etc)
    # which trigger on standard IIPs that are simply slightly wide or fast.
    MORPHOLOGICAL_FEATURES = [
        'M_plateau_25d',
        'gp_t_fall',
        'gp_t_rise',
        'gp_gr_slope',
        'residual_std',
        'lag1_autocorr',
        'tail_slope',
        'late_time_r2',
    ]

    @staticmethod
    def extract(batch_df: pd.DataFrame):
        """
        Step 6: The Aggregator.
        Runs Isolation Forest on PHYSICAL/MORPHOLOGICAL features only to avoid
        flagging standard IIPs that are merely parameter outliers.
        """
        if batch_df is None or len(batch_df) == 0:
            return batch_df

        # Use only morphological columns that actually exist in the batch
        feature_cols = [c for c in Aggregator.MORPHOLOGICAL_FEATURES if c in batch_df.columns]

        if not feature_cols:
            # Fallback: use all numeric cols minus IDs and REFITT params
            refitt_params = ['zams_final', 'mloss_rate_final', '56Ni_final',
                             'k_energy_final', 'beta_final', 'texp_final', 'A_v_final']
            exclude = ['object_id'] + refitt_params
            feature_cols = [c for c in batch_df.columns
                            if pd.api.types.is_numeric_dtype(batch_df[c])
                            and c not in exclude]

        # Impute NaNs with column median
        X = batch_df[feature_cols].apply(lambda x: x.fillna(x.median()), axis=0)

        if len(X) < 5:
            batch_df['population_deviation_score'] = 0
            batch_df['is_anomaly'] = False
            return batch_df

        iso = IsolationForest(contamination=0.1, random_state=42)
        preds = iso.fit_predict(X)
        batch_df['is_anomaly'] = preds == -1

        # Euclidean deviation from batch median in standardised feature space
        medians = X.median(axis=0)
        stds = X.std(axis=0) + 1e-9
        X_scaled = (X - medians) / stds
        batch_df['population_deviation_score'] = np.linalg.norm(X_scaled.values, axis=1)

        # Dynamic Benchmarking: predict M_plateau_25d from ZAMS + E_k scaling
        plateau_col = 'M_plateau_25d' if 'M_plateau_25d' in batch_df.columns else 'gp_M_peak'
        if plateau_col in batch_df.columns and 'zams_final' in batch_df.columns and 'k_energy_final' in batch_df.columns:
            from sklearn.linear_model import LinearRegression
            mask = (batch_df[plateau_col].notna()
                    & batch_df['zams_final'].notna()
                    & batch_df['k_energy_final'].notna())
            if mask.sum() > 5:
                reg = LinearRegression()
                X_train = batch_df.loc[mask, ['zams_final', 'k_energy_final']]
                y_train = batch_df.loc[mask, plateau_col]
                reg.fit(X_train, y_train)
                X_all = batch_df[['zams_final', 'k_energy_final']].fillna(
                    batch_df[['zams_final', 'k_energy_final']].median()
                )
                batch_df['M_peak_predicted'] = reg.predict(X_all)
                batch_df['M_peak_residual'] = batch_df[plateau_col] - batch_df['M_peak_predicted']
            else:
                batch_df['M_peak_predicted'] = np.nan
                batch_df['M_peak_residual'] = np.nan
                
        # Plateau Length Benchmarking: predict from (Mej^3 / E_k)^0.25 scaling proxy
        if 'plateau_duration_days' in batch_df.columns and 'zams_final' in batch_df.columns and 'k_energy_final' in batch_df.columns:
            mask = (batch_df['plateau_duration_days'].notna()
                    & batch_df['zams_final'].notna()
                    & batch_df['k_energy_final'].notna())
            
            if mask.sum() > 5:
                # Mej proxy = zams - 1.5
                mej_train = np.maximum(batch_df.loc[mask, 'zams_final'] - 1.5, 1.0)
                ek_train = batch_df.loc[mask, 'k_energy_final']
                X_train = ((mej_train ** 3) / ek_train) ** 0.25
                X_train = np.log10(X_train).values.reshape(-1, 1) # log linearize
                y_train = batch_df.loc[mask, 'plateau_duration_days'].values
                
                reg_plat = LinearRegression()
                reg_plat.fit(X_train, y_train)
                
                # predict for all
                mej_all = np.maximum(batch_df['zams_final'].fillna(batch_df['zams_final'].median()) - 1.5, 1.0)
                ek_all = batch_df['k_energy_final'].fillna(batch_df['k_energy_final'].median())
                X_all = ((mej_all ** 3) / ek_all) ** 0.25
                X_all = np.log10(X_all).values.reshape(-1, 1)
                
                batch_df['plateau_duration_predicted'] = reg_plat.predict(X_all)
                batch_df['plateau_duration_residual'] = batch_df['plateau_duration_days'] - batch_df['plateau_duration_predicted']
            else:
                batch_df['plateau_duration_predicted'] = np.nan
                batch_df['plateau_duration_residual'] = np.nan
        else:
            batch_df['plateau_duration_predicted'] = np.nan
            batch_df['plateau_duration_residual'] = np.nan

        return batch_df

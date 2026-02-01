#!/usr/bin/env python3
"""
Light Curve Completeness Validation for Type IIP Supernovae (SNCosmo-based)

Uses SNCosmo template fitting to determine light curve phase and completeness
status. This replaces custom physics heuristics with industry-standard templates.
"""

import numpy as np
import json
import sncosmo
from astropy.table import Table
from dataclasses import dataclass
from typing import Tuple, Optional
import warnings


@dataclass
class CompletenessScore:
    """SNCosmo-based completeness assessment for a SN IIP light curve."""
    
    # SNCosmo fit results
    t0_fitted: Optional[float]  # Explosion time (MJD)
    latest_phase: float  # Phase of latest observation (days since explosion)
    chi_squared_reduced: Optional[float]  # Goodness of fit
    template_name: str  # Which SNCosmo template was used
    fit_success: bool  # Whether template fitting succeeded
    
    # Phase classification
    phase_category: str  # Preliminary/Transitional/Validated
    
    # Overall status
    overall_status: str  # Incomplete/Partial/Validated
    
    def __str__(self):
        if self.fit_success:
            return (f"CompletenessScore(status={self.overall_status}, "
                    f"phase={self.latest_phase:.1f}d, χ²={self.chi_squared_reduced:.2f}, "
                    f"template={self.template_name})")
        else:
            return (f"CompletenessScore(status={self.overall_status}, "
                    f"phase_estimate={self.latest_phase:.1f}d, fit_failed)")


class LightCurveCompletenessChecker:
    """
    Validate Type IIP supernova light curve completeness using SNCosmo template fitting.
    
    This replaces custom heuristics with professional SN analysis tools.
    """
    
    # Phase boundaries for completeness (days since explosion)
    PHASE_PRELIMINARY = 70  # Too early for reliable convergence
    PHASE_TRANSITIONAL = 100  # Approaching completeness
    PHASE_VALIDATED = 100  # Confidently on radioactive tail
    
    # Template options (in order of preference for Type IIP)
    TEMPLATE_OPTIONS = ['nugent-sn2p', 's11-2005lc', 's11-2005hl']
    
    @staticmethod
    def _map_filter_name(filter_name: str) -> str:
        """
        Map filter names from data format to SNCosmo format.
        
        Examples: 'g-ztf' -> 'ztfg', 'r-ztf' -> 'ztfr'
        """
        filter_map = {
            'g-ztf': 'ztfg',
            'r-ztf': 'ztfr',
            'i-ztf': 'ztfi',
            'ztfg': 'ztfg',  # Already in correct format
            'ztfr': 'ztfr',
            'ztfi': 'ztfi',
        }
        return filter_map.get(filter_name, filter_name)
    
    def __init__(self, timeline_df=None, json_file=None):
        """
        Initialize with either timeline DataFrame or JSON file path.
        
        Args:
            timeline_df: DataFrame with columns ['Phase', 'filepath']
            json_file: Path to final observation JSON file
        """
        self.timeline_df = timeline_df
        self.json_file = json_file
        
    def check_completeness(self) -> CompletenessScore:
        """
        Run SNCosmo template fitting and determine completeness.
        
        Returns:
            CompletenessScore with fit results and validation status
        """
        # Load light curve data
        mjd, mags, mag_errs, filters = self._load_light_curve_data()
        
        if len(mjd) < 3:
            # Insufficient data for fitting
            return self._incomplete_score(0.0, fit_failed=True)
        
        # Try to fit SNCosmo template
        fit_result, template_name = self._fit_sncosmo_template(mjd, mags, mag_errs, filters)
        
        # Check if we should use SNCosmo results or fall back to Phase parameter
        use_sncosmo = False
        if fit_result is not None:
            # Check fit quality
            chi2_reduced = fit_result.chisq / fit_result.ndof if fit_result.ndof > 0 else 999
            t0 = fit_result.parameters[0]
            latest_phase = mjd[-1] - t0
            
            # Use SNCosmo only if fit is reasonable (chi2 < 20) and phase makes sense (> 0, < 500 days)
            if chi2_reduced < 20 and 0 < latest_phase < 500:
                use_sncosmo = True
        
        if use_sncosmo:
            # Use SNCosmo fitted phase
            phase_category = self.categorize_by_phase(latest_phase)
            overall_status = self._calculate_overall_status(latest_phase, chi2_reduced)
            
            return CompletenessScore(
                t0_fitted=t0,
                latest_phase=latest_phase,
                chi_squared_reduced=chi2_reduced,
                template_name=template_name,
                fit_success=True,
                phase_category=phase_category,
                overall_status=overall_status
            )
        else:
            # Fitting failed or unreliable - fall back to Phase parameter from data
            estimated_phase = self._estimate_phase_from_data()
            phase_category = self.categorize_by_phase(estimated_phase)
            overall_status = self._calculate_overall_status(estimated_phase, None)
            
            return CompletenessScore(
                t0_fitted=None,
                latest_phase=estimated_phase,
                chi_squared_reduced=None,
                template_name=template_name,
                fit_success=False,
                phase_category=phase_category,
                overall_status=overall_status
            )
    
    def _load_light_curve_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load photometric data in SNCosmo format.
        
        Returns:
            (mjd, mags, mag_errs, filters) arrays
        """
        if self.json_file:
            with open(self.json_file) as f:
                data = json.load(f)
            
            mjd_arr = np.array(data.get('mjd_arr', []))
            mag_arr_samples = data.get('mag_arr', [])
            mag_err_arr_samples = data.get('mag_err_arr', [])
            filter_name_raw = data.get('filter', data.get('Filter', 'r-ztf'))  # Try 'filter' first, then 'Filter'
            filter_name = self._map_filter_name(filter_name_raw)
            
            if len(mag_arr_samples) == 0 or len(mjd_arr) == 0:
                return np.array([]), np.array([]), np.array([]), np.array([])
            
            # Average over samples
            mags = np.mean(mag_arr_samples, axis=0)
            mag_errs = np.mean(mag_err_arr_samples, axis=0) if len(mag_err_arr_samples) > 0 else np.ones_like(mags) * 0.1
            
            # Create filter array
            filters = np.array([filter_name] * len(mjd_arr))
            
            return mjd_arr, mags, mag_errs, filters
            
        elif self.timeline_df is not None:
            mjd_list = []
            mag_list = []
            mag_err_list = []
            filter_list = []
            
            for _, row in self.timeline_df.iterrows():
                with open(row['filepath']) as f:
                    data = json.load(f)
                
                mjd_arr = np.array(data.get('mjd_arr', []))
                mag_arr = data.get('mag_arr', [])
                mag_err_arr = data.get('mag_err_arr', [])
                filter_name_raw = data.get('filter', data.get('Filter', 'r-ztf'))
                filter_name = self._map_filter_name(filter_name_raw)
                
                if len(mag_arr) > 0 and len(mjd_arr) > 0:
                    # Use mean magnitude at this observation
                    mag_mean = np.mean(mag_arr)
                    mag_err_mean = np.mean(mag_err_arr) if len(mag_err_arr) > 0 else 0.1
                    
                    mjd_list.extend(mjd_arr)
                    mag_list.extend([mag_mean] * len(mjd_arr))
                    mag_err_list.extend([mag_err_mean] * len(mjd_arr))
                    filter_list.extend([filter_name] * len(mjd_arr))
            
            return (np.array(mjd_list), np.array(mag_list), 
                   np.array(mag_err_list), np.array(filter_list))
        
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    def _fit_sncosmo_template(self, mjd: np.ndarray, mags: np.ndarray, 
                             mag_errs: np.ndarray, filters: np.ndarray) -> Tuple[Optional[object], str]:
        """
        Fit SNCosmo Type IIP template to light curve data.
        
        Args:
            mjd: Modified Julian Dates
            mags: Magnitudes
            mag_errs: Magnitude errors
            filters: Filter names
            
        Returns:
            (fit_result, template_name) or (None, template_name) if fit fails
        """
        # Create SNCosmo table (using astropy Table)
        # SNCosmo expects flux, not magnitudes - convert
        # Flux formula: flux = 10^(-0.4 * (mag - zp))
        zp = 25.0
        fluxes = 10**(-0.4 * (mags - zp))
        # Error propagation: dF/F = 0.4 * ln(10) * dmag
        flux_errs = fluxes * 0.4 * np.log(10) * mag_errs
        
        data_table = Table({
            'time': mjd,
            'band': filters,
            'flux': fluxes,
            'fluxerr': flux_errs,
            'zp': np.array([25.0] * len(mjd)),
            'zpsys': np.array(['ab'] * len(mjd))
        })
        
        # Try each template in order of preference
        for template_name in self.TEMPLATE_OPTIONS:
            try:
                # Create model
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = sncosmo.Model(source=template_name)
                
                # Set reasonable initial guesses
                t0_guess = mjd[0] - 20  # Assume explosion ~20 days before first observation
                model.set(t0=t0_guess, amplitude=1e-10)
                
                # Fit the model
                result, fitted_model = sncosmo.fit_lc(
                    data_table, model,
                    vparam_names=['t0', 'amplitude'],  # Fit explosion time and amplitude
                    bounds={'t0': (mjd[0] - 100, mjd[0]), 'amplitude': (0, 1e-8)}
                )
                
                # Check if fit is reasonable
                if result.success and result.parameters[0] < mjd[-1]:
                    return result, template_name
                    
            except Exception as e:
                # Try next template
                continue
        
        # All templates failed
        return None, self.TEMPLATE_OPTIONS[0]
    
    def _estimate_phase_from_data(self) -> float:
        """
        Fallback phase estimation when SNCosmo fitting fails.
        
        Uses the 'Phase' field from JSON if available.
        """
        if self.json_file:
            with open(self.json_file) as f:
                data = json.load(f)
            return data.get('parameters', {}).get('Phase', 0)
        
        elif self.timeline_df is not None and len(self.timeline_df) > 0:
            # Return the latest phase from timeline
            return self.timeline_df.iloc[-1].get('Phase', 0)
        
        return 0.0
    
    def categorize_by_phase(self, phase: float) -> str:
        """
        Categorize completeness by observation phase.
        
        Args:
            phase: Days since explosion
            
        Returns:
            "Preliminary" | "Transitional" | "Validated"
        """
        if phase < self.PHASE_PRELIMINARY:
            return "Preliminary"
        elif phase < self.PHASE_VALIDATED:
            return "Transitional"
        else:
            return "Validated"
    
    def _calculate_overall_status(self, latest_phase: float, 
                                  chi2_reduced: Optional[float]) -> str:
        """
        Determine overall completeness status from SNCosmo fit results.
        
        Criteria:
        - Validated: Phase > 100 days AND reasonable fit (χ² < 10)
        - Partial: Phase 70-100 days AND reasonable fit
        - Incomplete: Phase < 70 days OR poor fit
        
        Args:
            latest_phase: Days since explosion
            chi2_reduced: Reduced chi-squared from fit
            
        Returns:
            "Validated" | "Partial" | "Incomplete"
        """
        # Check fit quality
        fit_is_reasonable = (chi2_reduced is None or chi2_reduced < 10.0)
        
        # Validated: Late phase + good fit
        if latest_phase >= self.PHASE_VALIDATED and fit_is_reasonable:
            return "Validated"
        
        # Partial: Transitional phase + good fit
        if latest_phase >= self.PHASE_PRELIMINARY and latest_phase < self.PHASE_VALIDATED and fit_is_reasonable:
            return "Partial"
        
        # Incomplete: Early phase or bad fit
        return "Incomplete"
    
    def _incomplete_score(self, phase: float, fit_failed: bool = False) -> CompletenessScore:
        """Return an incomplete score for insufficient data or failed fits."""
        return CompletenessScore(
            t0_fitted=None,
            latest_phase=phase,
            chi_squared_reduced=None,
            template_name=self.TEMPLATE_OPTIONS[0],
            fit_success=not fit_failed,
            phase_category=self.categorize_by_phase(phase),
            overall_status="Incomplete"
        )


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 lightcurve_completeness.py <json_file>")
        sys.exit(1)
    
    checker = LightCurveCompletenessChecker(json_file=sys.argv[1])
    score = checker.check_completeness()
    
    print(f"\n{'='*70}")
    print("LIGHT CURVE COMPLETENESS ASSESSMENT (SNCosmo)")
    print(f"{'='*70}")
    print(f"Overall Status: {score.overall_status}")
    print(f"Latest Phase: {score.latest_phase:.1f} days since explosion")
    print(f"\nSNCosmo Template Fit:")
    print(f"  Template: {score.template_name}")
    
    if score.fit_success:
        print(f"  Explosion Time (t0): MJD {score.t0_fitted:.2f}")
        print(f"  Reduced χ²: {score.chi_squared_reduced:.2f}" if score.chi_squared_reduced else "  Reduced χ²: N/A")
        print(f"  Fit Quality: {'Good' if score.chi_squared_reduced and score.chi_squared_reduced < 5 else 'Acceptable' if score.chi_squared_reduced and score.chi_squared_reduced < 10 else 'Poor'}")
    else:
        print(f"  ⚠ Template fitting failed - using phase estimate from data")
    
    print(f"\nPhase Category: {score.phase_category}")
    
    # Interpretation
    print(f"\n{'Interpretation:':-^70}")
    if score.overall_status == "Validated":
        print("✓ Light curve is complete - observations extend to radioactive tail")
        print("  Parameter convergence is reliable for this object.")
    elif score.overall_status == "Partial":
        print("⚠ Light curve is approaching completeness")
        print("  Convergence metrics should be interpreted with caution.")
    else:
        print("✗ Light curve is incomplete - early phase or insufficient data")
        print("  May show false convergence. Metrics not reliable.")
    
    print(f"{'='*70}\n")

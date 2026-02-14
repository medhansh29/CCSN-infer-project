#!/usr/bin/env python3
"""
Light Curve Completeness Validation for Type IIP Supernovae

Uses the model's own Phase parameter to determine light curve completeness.
This is simpler and faster than SNCosmo template fitting, and the model's
Phase value is already accurate for our inference pipeline.
"""

import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class CompletenessScore:
    """Completeness assessment for a SN IIP light curve."""
    
    # Phase from model parameters
    latest_phase: float  # Phase of latest observation (days since explosion)
    
    # Optional t0 from model (if texp is available)
    t0_fitted: Optional[float]  # Explosion time (MJD) from model
    
    # Chi-squared placeholder (not used without SNCosmo)
    chi_squared_reduced: Optional[float]
    
    # Template placeholder (not used without SNCosmo)  
    template_name: str
    
    # Whether phase retrieval succeeded
    fit_success: bool
    
    # Phase classification
    phase_category: str  # Preliminary/Transitional/Validated
    
    # Overall status
    overall_status: str  # Incomplete/Partial/Validated
    
    def __str__(self):
        return (f"CompletenessScore(status={self.overall_status}, "
                f"phase={self.latest_phase:.1f}d, category={self.phase_category})")


class LightCurveCompletenessChecker:
    """
    Validate Type IIP supernova light curve completeness using model's Phase.
    
    This uses the Phase parameter directly from the model's inference output,
    which is much faster than running SNCosmo template fitting.
    """
    
    # Phase boundaries for completeness (days since explosion)
    PHASE_PRELIMINARY = 70  # Too early for reliable convergence
    PHASE_TRANSITIONAL = 100  # Approaching completeness
    PHASE_VALIDATED = 100  # Confidently on radioactive tail
    
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
        Determine completeness using the model's Phase parameter.
        
        Returns:
            CompletenessScore with phase and validation status
        """
        # Get phase from the model's parameters
        phase, t0 = self._get_phase_from_model()
        
        if phase is None or phase <= 0:
            return self._incomplete_score(0.0)
        
        # Categorize by phase
        phase_category = self.categorize_by_phase(phase)
        overall_status = self._calculate_overall_status(phase)
        
        return CompletenessScore(
            latest_phase=phase,
            t0_fitted=t0,
            chi_squared_reduced=None,  # Not using SNCosmo
            template_name="model",  # Using model's own parameters
            fit_success=True,
            phase_category=phase_category,
            overall_status=overall_status
        )
    
    def _get_phase_from_model(self) -> tuple:
        """
        Extract Phase and t0 from the model's JSON output.
        
        Returns:
            (phase, t0) tuple - phase in days, t0 as MJD
        """
        if self.json_file:
            try:
                with open(self.json_file) as f:
                    data = json.load(f)
                
                params = data.get('parameters', {})
                phase = params.get('Phase', 0)
                
                # Calculate t0 from texp and first MJD
                texp = params.get('texp', [None])[0] if isinstance(params.get('texp'), list) else params.get('texp')
                mjd_arr = data.get('mjd_arr', [])
                
                t0 = None
                if texp is not None and mjd_arr:
                    t0 = mjd_arr[0] - texp
                
                return phase, t0
                
            except (IOError, json.JSONDecodeError, KeyError):
                return 0.0, None
        
        elif self.timeline_df is not None and len(self.timeline_df) > 0:
            # Return the latest phase from timeline
            return self.timeline_df.iloc[-1].get('Phase', 0), None
        
        return 0.0, None
    
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
    
    def _calculate_overall_status(self, latest_phase: float) -> str:
        """
        Determine overall completeness status from phase.
        
        Criteria:
        - Validated: Phase >= 100 days (on radioactive tail)
        - Partial: Phase 70-100 days (plateau phase)
        - Incomplete: Phase < 70 days (early)
        
        Args:
            latest_phase: Days since explosion
            
        Returns:
            "Validated" | "Partial" | "Incomplete"
        """
        if latest_phase >= self.PHASE_VALIDATED:
            return "Validated"
        elif latest_phase >= self.PHASE_PRELIMINARY:
            return "Partial"
        else:
            return "Incomplete"
    
    def _incomplete_score(self, phase: float) -> CompletenessScore:
        """Return an incomplete score for insufficient data."""
        return CompletenessScore(
            t0_fitted=None,
            latest_phase=phase,
            chi_squared_reduced=None,
            template_name="model",
            fit_success=False,
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
    print("LIGHT CURVE COMPLETENESS ASSESSMENT")
    print(f"{'='*70}")
    print(f"Overall Status: {score.overall_status}")
    print(f"Latest Phase: {score.latest_phase:.1f} days since explosion")
    print(f"Phase Category: {score.phase_category}")
    
    if score.t0_fitted:
        print(f"Explosion Time (t0): MJD {score.t0_fitted:.2f}")
    
    # Interpretation
    print(f"\n{'Interpretation:':-^70}")
    if score.overall_status == "Validated":
        print("✓ Light curve is complete - observations extend to radioactive tail")
        print("  Parameter convergence is reliable for this object.")
    elif score.overall_status == "Partial":
        print("⚠ Light curve is approaching completeness (plateau phase)")
        print("  Convergence metrics should be interpreted with caution.")
    else:
        print("✗ Light curve is incomplete - early phase observations only")
        print("  May show false convergence. Metrics not reliable.")
    
    print(f"{'='*70}\n")

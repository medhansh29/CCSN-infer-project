#!/usr/bin/env python3
"""
Light Curve Completeness Validation for Type IIP Supernovae

Implements physical criteria to determine if a light curve is complete enough
for accurate parameter validation, avoiding "false convergence" from treating
the last observation as truth.

Based on Type IIP physics:
- Plateau phase (80-120 days)
- Plateau drop-off (slope steepening)  
- Radioactive tail (Co-56 decay at 0.0098 mag/day)
"""

import numpy as np
import json
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class CompletenessScore:
    """Physical completeness assessment for a SN IIP light curve."""
    
    # Individual criteria
    has_plateau_dropoff: bool
    on_radioactive_tail: bool
    phase_category: str  # Preliminary/Transitional/Validated
    sufficient_dimming: bool
    
    # Quantitative measurements
    total_dimming_mag: float
    tail_slope: Optional[float]
    dropoff_phase: Optional[float]
    final_phase: float
    
    # Overall status
    overall_status: str  # Incomplete/Partial/Validated
    
    def __str__(self):
        return (f"CompletenessScore(status={self.overall_status}, "
                f"phase={self.final_phase:.1f}d, dimming={self.total_dimming_mag:.2f}mag)")


class LightCurveCompletenessChecker:
    """
    Validate Type IIP supernova light curve completeness using physical criteria.
    
    Implements four methods:
    1. Slope-Break Method: Detect plateau drop-off
    2. Radioactive Tail Alignment: Verify Co-56 decay slope
    3. Phase-Window Filter: Temporal validation
    4. Flux Ratio Threshold: Dynamic range check
    """
    
    # Physical constants
    CO56_DECAY_SLOPE = 0.0098  # mag/day
    CO56_SLOPE_TOLERANCE = 0.002  # ±0.002 mag/day
    PLATEAU_DROPOFF_THRESHOLD = 0.1  # mag/day
    MIN_DROPOFF_DURATION = 10  # days
    MIN_DIMMING_FOR_VALIDATION = 2.5  # magnitudes
    
    # Phase boundaries
    PHASE_PRELIMINARY = 70  # days
    PHASE_TRANSITIONAL = 120  # days
    
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
        Run all completeness checks and return overall score.
        
        Returns:
            CompletenessScore with all validation results
        """
        # Load light curve data
        phases, mags = self._load_light_curve()
        
        if len(phases) < 3:
            return self._incomplete_score(phases[-1] if len(phases) > 0 else 0)
        
        # Method 1: Plateau drop-off detection
        has_dropoff, dropoff_phase = self.detect_plateau_dropoff(phases, mags)
        
        # Method 2: Radioactive tail check
        on_tail, tail_slope = self.check_radioactive_tail(phases, mags)
        
        # Method 3: Phase categorization
        phase_category = self.categorize_by_phase(phases[-1])
        
        # Method 4: Dynamic range
        sufficient_dimming, total_dimming = self.check_dynamic_range(mags)
        
        # Determine overall status
        overall_status = self._calculate_overall_status(
            has_dropoff, on_tail, phase_category, sufficient_dimming
        )
        
        return CompletenessScore(
            has_plateau_dropoff=has_dropoff,
            on_radioactive_tail=on_tail,
            phase_category=phase_category,
            sufficient_dimming=sufficient_dimming,
            total_dimming_mag=total_dimming,
            tail_slope=tail_slope,
            dropoff_phase=dropoff_phase,
            final_phase=phases[-1],
            overall_status=overall_status
        )
    
    def _load_light_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load phase and magnitude arrays from timeline or JSON."""
        if self.json_file:
            with open(self.json_file) as f:
                data = json.load(f)
            
            mjd_arr = np.array(data.get('mjd_arr', []))
            mag_arr_samples = data.get('mag_arr', [])
            
            if len(mag_arr_samples) == 0 or len(mjd_arr) == 0:
                return np.array([]), np.array([])
            
            # Average over samples
            mag_arr = np.mean(mag_arr_samples, axis=0)
            
            # Convert to phase
            explosion_mjd = mjd_arr[0] - data.get('parameters', {}).get('Phase', 0)
            phases = mjd_arr - explosion_mjd
            
            return phases, mag_arr
            
        elif self.timeline_df is not None:
            phases = []
            mags = []
            
            for _, row in self.timeline_df.iterrows():
                with open(row['filepath']) as f:
                    data = json.load(f)
                
                phase = data.get('parameters', {}).get('Phase', 0)
                mag_arr = data.get('mag_arr', [])
                
                if len(mag_arr) > 0:
                    # Use mean magnitude at this observation
                    mag_mean = np.mean(mag_arr)
                    phases.append(phase)
                    mags.append(mag_mean)
            
            return np.array(phases), np.array(mags)
        
        return np.array([]), np.array([])
    
    # ========================================================================
    # Method 1: Slope-Break Detection (Plateau Drop-Off)
    # ========================================================================
    
    def detect_plateau_dropoff(self, phases: np.ndarray, mags: np.ndarray) -> Tuple[bool, Optional[float]]:
        """
        Detect the steepening event marking the end of plateau phase.
        
        The plateau drop-off is characterized by:
        - Slope exceeding ~0.1 mag/day for 10+ days
        - Followed by flattening to radioactive tail
        
        Args:
            phases: Array of phases (days since explosion)
            mags: Array of magnitudes
            
        Returns:
            (has_dropoff, dropoff_phase) where dropoff_phase is when it occurs
        """
        if len(phases) < 5:
            return False, None
        
        # Calculate rolling slopes with 5-point window
        window = 5
        slopes = []
        slope_phases = []
        
        for i in range(len(phases) - window + 1):
            phase_window = phases[i:i+window]
            mag_window = mags[i:i+window]
            
            # Linear fit
            if len(phase_window) > 1:
                slope = np.polyfit(phase_window, mag_window, 1)[0]
                slopes.append(abs(slope))  # Use absolute value
                slope_phases.append(np.mean(phase_window))
        
        slopes = np.array(slopes)
        slope_phases = np.array(slope_phases)
        
        # Look for sustained steep slope
        steep_mask = slopes > self.PLATEAU_DROPOFF_THRESHOLD
        
        if not np.any(steep_mask):
            return False, None
        
        # Find continuous steep regions
        steep_indices = np.where(steep_mask)[0]
        
        if len(steep_indices) == 0:
            return False, None
        
        # Check if steep phase lasts long enough
        steep_phases = slope_phases[steep_indices]
        if len(steep_phases) > 0 and (steep_phases[-1] - steep_phases[0]) >= self.MIN_DROPOFF_DURATION:
            # Found a sustained drop-off
            dropoff_phase = steep_phases[0]
            return True, dropoff_phase
        
        return False, None
    
    # ========================================================================
    # Method 2: Radioactive Tail Alignment
    # ========================================================================
    
    def check_radioactive_tail(self, phases: np.ndarray, mags: np.ndarray) -> Tuple[bool, Optional[float]]:
        """
        Verify if light curve has reached Co-56 radioactive decay tail.
        
        The tail has a very specific slope of 0.0098 ± 0.002 mag/day from Co-56 
        decay physics.
        
        Args:
            phases: Array of phases
            mags: Array of magnitudes
            
        Returns:
            (on_tail, tail_slope) where tail_slope is measured value
        """
        if len(phases) < 15:
            return False, None
        
        # Fit linear slope to last 15-20 points
        n_points = min(20, len(phases))
        late_phases = phases[-n_points:]
        late_mags = mags[-n_points:]
        
        # Linear fit
        slope, intercept = np.polyfit(late_phases, late_mags, 1)
        
        # Check if slope matches Co-56 decay
        expected_slope = self.CO56_DECAY_SLOPE
        slope_diff = abs(slope - expected_slope)
        
        on_tail = slope_diff < self.CO56_SLOPE_TOLERANCE
        
        return on_tail, slope
    
    # ========================================================================
    # Method 3: Phase-Window Filter
    # ========================================================================
    
    def categorize_by_phase(self, final_phase: float) -> str:
        """
        Categorize completeness by observation window.
        
        Args:
            final_phase: Final observation phase (days)
            
        Returns:
            "Preliminary" | "Transitional" | "Validated"
        """
        if final_phase < self.PHASE_PRELIMINARY:
            return "Preliminary"
        elif final_phase < self.PHASE_TRANSITIONAL:
            return "Transitional"
        else:
            return "Validated"
    
    # ========================================================================
    # Method 4: Flux Ratio Threshold
    # ========================================================================
    
    def check_dynamic_range(self, mags: np.ndarray) -> Tuple[bool, float]:
        """
        Verify sufficient dimming from peak brightness.
        
        Requires ≥2.5 mag dimming to ensure model has seen the information
        gain from recombination phase ending.
        
        Args:
            mags: Array of magnitudes
            
        Returns:
            (sufficient_range, total_dimming)
        """
        if len(mags) < 2:
            return False, 0.0
        
        # Calculate dimming (mags get larger = dimmer)
        mag_peak = np.min(mags)  # Brightest = smallest mag
        mag_current = mags[-1]
        
        total_dimming = mag_current - mag_peak
        
        sufficient = total_dimming >= self.MIN_DIMMING_FOR_VALIDATION
        
        return sufficient, total_dimming
    
    # ========================================================================
    # Overall Status Calculation
    # ========================================================================
    
    def _calculate_overall_status(self, has_dropoff: bool, on_tail: bool, 
                                  phase_category: str, sufficient_dimming: bool) -> str:
        """
        Combine all criteria to determine overall completeness status.
        
        Validated: All critical criteria met OR (tail + phase > 120)
        Partial: Some criteria met, phase > 70
        Incomplete: Early phase or missing key features
        """
        # Strictest validation: all criteria
        if has_dropoff and on_tail and sufficient_dimming and phase_category == "Validated":
            return "Validated"
        
        # Relaxed validation: on tail with mature phase
        if on_tail and phase_category == "Validated":
            return "Validated"
        
        # Partial validation: transitional phase with some features
        if phase_category in ["Transitional", "Validated"] and (on_tail or has_dropoff):
            return "Partial"
        
        # Everything else is incomplete
        return "Incomplete"
    
    def _incomplete_score(self, phase: float) -> CompletenessScore:
        """Return an incomplete score for early/insufficient data."""
        return CompletenessScore(
            has_plateau_dropoff=False,
            on_radioactive_tail=False,
            phase_category=self.categorize_by_phase(phase),
            sufficient_dimming=False,
            total_dimming_mag=0.0,
            tail_slope=None,
            dropoff_phase=None,
            final_phase=phase,
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
    print(f"Final Phase: {score.final_phase:.1f} days")
    print(f"\nIndividual Criteria:")
    print(f"  ✓ Plateau Drop-off: {'YES' if score.has_plateau_dropoff else 'NO'}")
    if score.dropoff_phase:
        print(f"    → Detected at {score.dropoff_phase:.1f} days")
    print(f"  ✓ Radioactive Tail: {'YES' if score.on_radioactive_tail else 'NO'}")
    if score.tail_slope:
        print(f"    → Slope: {score.tail_slope:.4f} mag/day (expected: 0.0098)")
    print(f"  ✓ Phase Category: {score.phase_category}")
    print(f"  ✓ Sufficient Dimming: {'YES' if score.sufficient_dimming else 'NO'}")
    print(f"    → Total dimming: {score.total_dimming_mag:.2f} mag")
    print(f"{'='*70}\n")

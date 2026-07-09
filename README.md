# Kepler Light Curve Period Analysis

This repo analyzes one Kepler light curve from a FITS file. The script reads the flux data with Astropy, removes flagged points, normalizes the light curve, searches for a dominant period, and checks the result with a simple sinusoidal fit.

The current run uses the Kepler FITS file included in the repo:

    kplr000757450-2009350155506_llc.fits

## What the script does

The analysis starts with the Kepler table in the FITS file. It uses `PDCSAP_FLUX` when that column is available, then removes points with invalid time or flux values, nonpositive flux, or nonzero `QUALITY` flags. The removed rows are saved separately so the cleaning step can be checked later.

After cleaning, the flux is divided by its median value. The script then runs a Lomb-Scargle period search over trial periods from 0.5 to 40 days. A sinusoidal curve is fit near the strongest periodogram peak. The fit is not meant to prove the physical source of the variability. It is a compact model used for checking the period and plotting residuals.

## Current outputs

Running the script creates:

    outputs/cleaned_light_curve.csv
    outputs/flagged_removed_points.csv
    outputs/period_analysis_summary.csv
    outputs/cleaned_light_curve.png
    outputs/lomb_scargle_periodogram.png
    outputs/model_fit.png
    outputs/phase_folded_light_curve.png
    outputs/residuals.png

The summary CSV records the FITS file used, the flux column used, the number of removed points, the Lomb-Scargle period, an approximate period error from the peak width, the fitted period, and residual statistics.

## Plots

Cleaned light curve:

![Cleaned light curve](outputs/cleaned_light_curve.png)

Lomb-Scargle period search:

![Lomb-Scargle periodogram](outputs/lomb_scargle_periodogram.png)

Sinusoidal fit:

![Model fit](outputs/model_fit.png)

Phase-folded light curve:

![Phase-folded light curve](outputs/phase_folded_light_curve.png)

Residuals:

![Residuals](outputs/residuals.png)

## Notes

Kepler light curves can show periodic structure for several reasons, including stellar rotation, pulsation, eclipsing systems, or transits. This repo does not claim a planet detection. It is a period-analysis workflow for a real Kepler FITS light curve.

The period error reported here is a simple diagnostic from the width of the periodogram peak. It should not be treated as a full statistical uncertainty from a physical light-curve model.

## Run

Install the dependencies:

    python3 -m pip install numpy pandas matplotlib astropy scipy

Run the script:

    python3 src/main.py

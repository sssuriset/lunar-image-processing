import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit


def find_file():
    files = []
    for pattern in ["data/*.fits", "data/*.fit", "*.fits", "*.fit"]:
        files.extend(glob.glob(pattern))

    if not files:
        raise FileNotFoundError("No Kepler FITS file found.")

    return files[0]


def read_curve(path):
    with fits.open(path) as hdul:
        tab = hdul[1].data
        cols = tab.columns.names

        if "TIME" not in cols:
            raise ValueError("Missing TIME column.")

        time = np.asarray(tab["TIME"], dtype=float)

        if "PDCSAP_FLUX" in cols:
            flux = np.asarray(tab["PDCSAP_FLUX"], dtype=float)
            source = "PDCSAP_FLUX"
        elif "SAP_FLUX" in cols:
            flux = np.asarray(tab["SAP_FLUX"], dtype=float)
            source = "SAP_FLUX"
        else:
            raise ValueError("Missing flux column.")

        if "PDCSAP_FLUX_ERR" in cols:
            err = np.asarray(tab["PDCSAP_FLUX_ERR"], dtype=float)
        elif "SAP_FLUX_ERR" in cols:
            err = np.asarray(tab["SAP_FLUX_ERR"], dtype=float)
        else:
            err = np.full_like(flux, np.nan)

        if "QUALITY" in cols:
            quality = np.asarray(tab["QUALITY"], dtype=int)
        else:
            quality = np.zeros_like(time, dtype=int)

    raw_n = len(time)

    keep = np.isfinite(time) & np.isfinite(flux) & (flux > 0) & (quality == 0)

    clean = pd.DataFrame({
        "time_bkjd": time[keep],
        "flux": flux[keep],
        "flux_error": err[keep],
        "quality": quality[keep],
    })

    removed = pd.DataFrame({
        "time_bkjd": time[~keep],
        "flux": flux[~keep],
        "flux_error": err[~keep],
        "quality": quality[~keep],
    })

    clean["norm_flux"] = clean["flux"] / np.nanmedian(clean["flux"])

    return clean, removed, raw_n, source


def wave(t, offset, amp, period, phase):
    return offset + amp * np.sin(2 * np.pi * t / period + phase)


def periodogram(time, flux, pmin=0.5, pmax=40):
    y = flux - np.nanmean(flux)
    freq = np.linspace(1 / pmax, 1 / pmin, 25000)

    ls = LombScargle(time, y)
    power = ls.power(freq)

    periods = 1 / freq
    i = np.argmax(power)

    false_alarm = float(np.asarray(ls.false_alarm_level(0.01)).mean())

    return periods, power, periods[i], power[i], false_alarm


def peak_width(periods, power):
    i = np.argmax(power)
    base = np.nanmedian(power)
    half = base + 0.5 * (power[i] - base)

    left = i
    right = i

    while left > 0 and power[left] > half:
        left -= 1

    while right < len(power) - 1 and power[right] > half:
        right += 1

    lo = periods[min(left, right)]
    hi = periods[max(left, right)]
    err = abs(hi - lo) / 2

    if err == 0 or not np.isfinite(err):
        err = np.nan

    return err, lo, hi


def fit_wave(time, flux, guess):
    amp_guess = 0.5 * (np.nanmax(flux) - np.nanmin(flux))

    start = [1.0, amp_guess, guess, 0.0]
    bounds = (
        [0.5, -1.0, 0.75 * guess, -2 * np.pi],
        [1.5, 1.0, 1.25 * guess, 2 * np.pi],
    )

    params, cov = curve_fit(
        wave,
        time,
        flux,
        p0=start,
        bounds=bounds,
        maxfev=20000,
    )

    model = wave(time, *params)
    resid = flux - model

    if cov is not None and np.isfinite(cov[2, 2]):
        perr = np.sqrt(cov[2, 2])
    else:
        perr = np.nan

    return params, model, resid, params[2], perr


def fold(time, flux, period):
    phase = (time % period) / period
    order = np.argsort(phase)
    return phase[order], flux[order]


def save_curve(data):
    plt.figure(figsize=(10, 5))
    plt.scatter(data["time_bkjd"], data["norm_flux"], s=5, alpha=0.55)
    plt.xlabel("Time (BKJD)")
    plt.ylabel("Normalized flux")
    plt.title("Cleaned Kepler light curve")
    plt.tight_layout()
    plt.savefig("outputs/cleaned_light_curve.png", dpi=300)
    plt.close()


def save_power(periods, power, best, false_alarm):
    plt.figure(figsize=(10, 5))
    plt.plot(periods, power, linewidth=1)
    plt.axvline(best, linestyle="--", label=f"{best:.4f} days")
    plt.axhline(false_alarm, linestyle=":", label="1% false alarm level")
    plt.xlabel("Period (days)")
    plt.ylabel("Lomb-Scargle power")
    plt.title("Period search")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/lomb_scargle_periodogram.png", dpi=300)
    plt.close()


def save_fit(time, flux, model):
    plt.figure(figsize=(10, 5))
    plt.scatter(time, flux, s=5, alpha=0.45, label="Data")
    plt.plot(time, model, linewidth=1.5, label="Sinusoidal fit")
    plt.xlabel("Time (BKJD)")
    plt.ylabel("Normalized flux")
    plt.title("Light curve model")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/model_fit.png", dpi=300)
    plt.close()


def save_folded(time, flux, model, period):
    phase, f = fold(time, flux, period)
    model_phase, m = fold(time, model, period)

    plt.figure(figsize=(10, 5))
    plt.scatter(phase, f, s=5, alpha=0.45, label="Data")
    plt.plot(model_phase, m, linewidth=2, label="Fit")
    plt.xlabel("Phase")
    plt.ylabel("Normalized flux")
    plt.title("Phase-folded light curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/phase_folded_light_curve.png", dpi=300)
    plt.close()


def save_resid(time, resid):
    plt.figure(figsize=(10, 5))
    plt.axhline(0, linestyle="--")
    plt.scatter(time, resid, s=5, alpha=0.5)
    plt.xlabel("Time (BKJD)")
    plt.ylabel("Residual flux")
    plt.title("Fit residuals")
    plt.tight_layout()
    plt.savefig("outputs/residuals.png", dpi=300)
    plt.close()


def main():
    os.makedirs("outputs", exist_ok=True)

    path = find_file()
    data, removed, raw_n, source = read_curve(path)

    time = data["time_bkjd"].to_numpy()
    flux = data["norm_flux"].to_numpy()

    periods, power, ls_period, ls_power, false_alarm = periodogram(time, flux)
    ls_err, p_lo, p_hi = peak_width(periods, power)

    params, model, resid, fit_period, fit_err = fit_wave(time, flux, ls_period)

    data["model_flux"] = model
    data["residual_flux"] = resid

    data.to_csv("outputs/cleaned_light_curve.csv", index=False)
    removed.to_csv("outputs/flagged_removed_points.csv", index=False)

    rms = np.sqrt(np.mean(resid**2))
    std = np.std(resid, ddof=1)

    summary = pd.DataFrame({
        "metric": [
            "fits_file",
            "flux_column",
            "raw_points",
            "clean_points",
            "removed_points",
            "lomb_scargle_period_days",
            "lomb_scargle_period_error_days",
            "lomb_scargle_power",
            "false_alarm_level_1pct",
            "fit_period_days",
            "fit_period_error_days",
            "period_lower_bound_days",
            "period_upper_bound_days",
            "residual_rms",
            "residual_std",
        ],
        "value": [
            path,
            source,
            raw_n,
            len(data),
            len(removed),
            ls_period,
            ls_err,
            ls_power,
            false_alarm,
            fit_period,
            fit_err,
            p_lo,
            p_hi,
            rms,
            std,
        ],
    })

    summary.to_csv("outputs/period_analysis_summary.csv", index=False)

    save_curve(data)
    save_power(periods, power, ls_period, false_alarm)
    save_fit(time, flux, model)
    save_folded(time, flux, model, fit_period)
    save_resid(time, resid)

    print("FITS file:", path)
    print("Flux column:", source)
    print("Raw points:", raw_n)
    print("Clean points:", len(data))
    print("Removed points:", len(removed))
    print("Lomb-Scargle period:", round(ls_period, 5), "days")
    print("Lomb-Scargle period error:", round(ls_err, 5), "days")
    print("Fit period:", round(fit_period, 5), "days")
    print("Fit period error:", round(fit_err, 5), "days")
    print("Residual RMS:", round(rms, 6))


if __name__ == "__main__":
    main()

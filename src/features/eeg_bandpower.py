import numpy as np
import pandas as pd
import mne

DEFAULT_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 15.0),
    "beta":  (15.0, 30.0),
}

def compute_eeg_bandpower(
    epochs,
    picks=None,
    bands=DEFAULT_BANDS,
    psd_method="welch",   # "welch" or "multitaper"
    fmin=0.5,
    fmax=30.0,
    relative=True,
):
    
    if picks is None:
        picks = mne.pick_types(epochs.info, eeg=True, exclude="bads")

    psd = epochs.compute_psd(
        method=psd_method,
        fmin=fmin,
        fmax=fmax,
        picks=picks,
        verbose="ERROR",
    )
    freqs = psd.freqs
    pxx = psd.get_data()

    if all(isinstance(p, int) for p in picks):
        ch_names = [epochs.ch_names[i] for i in picks]
    else:
        ch_names = list(picks)

    total_power = np.trapezoid(pxx, freqs, axis=-1)

    features = {}

    for band_name, (lo, hi) in bands.items():
        band_mask = (freqs >= lo) & (freqs < hi)
        if not np.any(band_mask):
            continue

        band_freqs = freqs[band_mask]
        band_pxx = pxx[:, :, band_mask]

        band_power = np.trapezoid(band_pxx, band_freqs, axis=-1)

        if relative:
            band_power = band_power / np.maximum(total_power, 1e-12)

        for ch_i, ch in enumerate(ch_names):
            features[f"{ch}__{band_name}"] = band_power[:, ch_i]

    return pd.DataFrame(features)

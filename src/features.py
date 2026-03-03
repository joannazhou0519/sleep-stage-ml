import numpy as np
import pandas as pd
import mne
import librosa
import neurokit2 as nk

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
    *,
    reduce_channels="mean",   # None | "mean" | "sum"
    return_array=False,
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
    pxx = psd.get_data()  # (n_epochs, n_channels, n_freqs)

    if all(isinstance(p, int) for p in picks):
        ch_names = [epochs.ch_names[i] for i in picks]
    else:
        ch_names = list(picks)

    # Total power for "relative" normalization
    total_power = np.trapezoid(pxx, freqs, axis=-1)  # (n_epochs, n_channels)

    band_names = []
    band_power_list = []  # list of (n_epochs, n_channels)
    for band_name, (lo, hi) in bands.items():
        band_mask = (freqs >= lo) & (freqs < hi)
        if not np.any(band_mask):
            continue

        band_freqs = freqs[band_mask]
        band_pxx = pxx[:, :, band_mask]

        band_power = np.trapezoid(band_pxx, band_freqs, axis=-1)  # (n_epochs, n_channels)

        if relative:
            band_power = band_power / np.maximum(total_power, 1e-12)

        band_names.append(band_name)
        band_power_list.append(band_power)

    if len(band_power_list) == 0:
        if return_array:
            return np.zeros((len(epochs), 0), dtype=float), []
        return pd.DataFrame(index=np.arange(len(epochs)))

    # (n_epochs, n_channels, n_bands)
    bp = np.stack(band_power_list, axis=-1)

    if reduce_channels is not None:
        reduce_channels = str(reduce_channels).lower()
        if reduce_channels == "mean":
            X = bp.mean(axis=1)  # (n_epochs, n_bands)
        elif reduce_channels == "sum":
            X = bp.sum(axis=1)   # (n_epochs, n_bands)
        else:
            raise ValueError("reduce_channels must be None, 'mean', or 'sum'")

        feature_names = list(band_names)

        if return_array:
            return X, feature_names
        return pd.DataFrame(X, columns=feature_names)

    n_epochs = bp.shape[0]
    n_ch = bp.shape[1]
    n_b = bp.shape[2]

    X = np.transpose(bp, (0, 2, 1)).reshape(n_epochs, n_b * n_ch)
    feature_names = [f"{band}__{ch}" for band in band_names for ch in ch_names]

    if return_array:
        return X, feature_names
    return pd.DataFrame(X, columns=feature_names)


def compute_eeg_spectrogram(
    epochs,
    picks=None,
    *,
    n_fft=256,
    hop_length=64,
    win_length=None,
    fmin=0.5,
    fmax=30.0,
    log_base=True,
    eps=1e-10,
    center=False,
):

    X = epochs.get_data(picks=picks)  # (N, C, T)
    sfreq = float(epochs.info["sfreq"])

    S_all = []

    for ep in range(X.shape[0]):
        ch_specs = []

        for ch in range(X.shape[1]):
            y = X[ep, ch].astype(np.float32, copy=False)

            Z = librosa.stft(
                y,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window="hann",
                center=center,
            )

            S = np.abs(Z) ** 2

            freqs = librosa.fft_frequencies(sr=sfreq, n_fft=n_fft)

            mask = (freqs >= fmin) & (freqs <= fmax)
            S = S[mask]

            if (log_base):
                S = 10 * np.log10(np.maximum(S, eps))
            
            ch_specs.append(S)

        S_all.append(np.stack(ch_specs, axis=0))

    S_all = np.stack(S_all, axis=0)

    return S_all


def compute_ppg_features(ppg_signal, sampling_rate, epoch_length=30):

    ppg_signal = np.asarray(ppg_signal, dtype=float)

    samples_per_epoch = int(epoch_length * sampling_rate)
    n_epochs = len(ppg_signal) // samples_per_epoch

    rows = []

    for epoch in range(n_epochs):
        a = epoch * samples_per_epoch
        b = a + samples_per_epoch
        seg = ppg_signal[a:b]

        row = {"epoch": epoch, "bad_epoch": False, "bad_reason": ""}

        if not np.all(np.isfinite(seg)):
            row["bad_epoch"] = True
            row["bad_reason"] = "nan_or_inf"
            rows.append(row)
            continue

        if np.std(seg) < 1e-6:
            row["bad_epoch"] = True
            row["bad_reason"] = "flat_signal"
            rows.append(row)
            continue

        seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-12)

        try:
            seg_clean = nk.ppg_clean(seg, sampling_rate=sampling_rate)


            peaks_dict, _ = nk.ppg_peaks(seg_clean, sampling_rate=sampling_rate)
            peaks = peaks_dict.get("PPG_Peaks", None)

            if peaks is None:
                row["bad_epoch"] = True
                row["bad_reason"] = "no_peaks"
                rows.append(row)
                continue

            peak_idx = np.where(peaks == 1)[0]


            if len(peak_idx) < 5:
                row["bad_epoch"] = True
                row["bad_reason"] = "too_few_peaks"
                rows.append(row)
                continue

            hrv = nk.hrv_time(peak_idx, sampling_rate=sampling_rate, show=False)

            row["RMSSD"] = float(hrv["HRV_RMSSD"].iloc[0])
            row["SDNN"] = float(hrv["HRV_SDNN"].iloc[0])

            ibi_ms = np.diff(peak_idx) / sampling_rate * 1000.0
            mean_ibi = float(np.mean(ibi_ms))
            row["HR_mean_bpm"] = 60000.0 / mean_ibi if mean_ibi > 0 else np.nan
            row["n_beats"] = int(len(peak_idx))

            # Pulse amplitude variability proxy (on cleaned normalized waveform)
            peak_vals = seg_clean[peaks]
            row["pulse_amp_mean"] = float(np.mean(peak_vals))
            row["pulse_amp_std"] = float(np.std(peak_vals))
            
        except Exception as e:
            row["bad_epoch"] = True
            row["bad_reason"] = f"exception:{type(e).__name__}"

        rows.append(row)

    return pd.DataFrame(rows)


import numpy as np
import pandas as pd
import mne
import librosa

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

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
    stack_order="band_then_channel",  # "band_then_channel" or "channel_then_band"
    return_array=False,
):
    """
    EEG bandpower from MNE Epochs.

    Modes
    -----
    1) reduce_channels in {"mean","sum"}:
         Output features: (n_epochs, n_bands)
         Feature names: ["delta","theta",...]
         - "sum": adds all channels' bandpower for each band
         - "mean": averages across channels for each band

    2) reduce_channels=None:
         Output features: (n_epochs, n_channels*n_bands)
         Ordering controlled by stack_order:
           - "band_then_channel": [delta ch1..chN, theta ch1..chN, ...]
           - "channel_then_band": [ch1 delta..beta, ch2 delta..beta, ...]

    Returns
    -------
    If return_array=True:
        (X, feature_names)
    else:
        pd.DataFrame
    """
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

    if stack_order == "band_then_channel":
        # (n_epochs, n_channels, n_bands) -> (n_epochs, n_bands, n_channels) -> flatten
        X = np.transpose(bp, (0, 2, 1)).reshape(n_epochs, n_b * n_ch)
        feature_names = [f"{band}__{ch}" for band in band_names for ch in ch_names]
    elif stack_order == "channel_then_band":
        X = bp.reshape(n_epochs, n_ch * n_b)
        feature_names = [f"{ch}__{band}" for ch in ch_names for band in band_names]
    else:
        raise ValueError("stack_order must be 'band_then_channel' or 'channel_then_band'")

    if return_array:
        return X, feature_names
    return pd.DataFrame(X, columns=feature_names)


def compute_logmel_spectrogram(
    epochs,
    picks=None,
    *,
    n_mels=64,
    n_fft=256,
    hop_length=64,
    win_length=None,
    fmin=0.5,
    fmax=30.0,
    log_base=10.0,
    eps=1e-10,
    stack_mode="channels",  # "channels" | "concat_freq" | "concat_time"
    center=False,
    pad_mode="reflect",
):
    """
    Log-mel spectrogram per epoch from MNE Epochs.

    Returns S, meta.

    Shapes by stack_mode (PyTorch-friendly):
      - "channels":   S shape (n_epochs, n_channels, n_mels, n_frames)
         Use Conv2d with input channels = n_channels.

      - "concat_freq": S shape (n_epochs, 1, n_mels*n_channels, n_frames)
         Think: stack mel bins across channels into a taller "image".

      - "concat_time": S shape (n_epochs, 1, n_mels, n_frames*n_channels)
         Think: place each channel's spectrogram side-by-side in time.

    Notes:
      - Choose n_fft / hop_length based on sfreq.
    """

    if picks is None:
        picks = mne.pick_types(epochs.info, eeg=True, exclude="bads")

    X = epochs.get_data(picks=picks)  # (n_epochs, n_channels, n_times)
    sfreq = float(epochs.info["sfreq"])

    if all(isinstance(p, int) for p in picks):
        ch_names = [epochs.ch_names[i] for i in picks]
    else:
        ch_names = list(picks)

    if win_length is None:
        win_length = n_fft

    S_all = []
    for ep in range(X.shape[0]):
        ch_mels = []
        for ch in range(X.shape[1]):
            y = X[ep, ch].astype(np.float32, copy=False)

            mel = librosa.feature.melspectrogram(
                y=y,
                sr=sfreq,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window="hann",
                center=center,
                pad_mode=pad_mode,
                power=2.0,
                n_mels=n_mels,
                fmin=fmin,
                fmax=fmax,
            )  # (n_mels, n_frames)

            if log_base == 10.0:
                mel_log = 10.0 * np.log10(np.maximum(mel, eps))
            elif log_base == np.e:
                mel_log = np.log(np.maximum(mel, eps))
            else:
                mel_log = np.log(np.maximum(mel, eps)) / np.log(float(log_base))

            ch_mels.append(mel_log)

        S_all.append(np.stack(ch_mels, axis=0))  # (n_channels, n_mels, n_frames)

    S = np.stack(S_all, axis=0)  # (n_epochs, n_channels, n_mels, n_frames)

    stack_mode = str(stack_mode).lower()
    if stack_mode == "channels":
        # already in PyTorch Conv2d format (N, C, H, W) if you treat mel as H and frames as W
        S_out = S
    elif stack_mode == "concat_freq":
        # (N, C, M, T) -> (N, 1, M*C, T)
        S_out = np.transpose(S, (0, 2, 1, 3)).reshape(S.shape[0], 1, n_mels * len(ch_names), S.shape[-1])
    elif stack_mode == "concat_time":
        # (N, C, M, T) -> (N, 1, M, T*C)
        S_out = np.transpose(S, (0, 2, 3, 1)).reshape(S.shape[0], 1, n_mels, S.shape[-1] * len(ch_names))
    else:
        raise ValueError("stack_mode must be 'channels', 'concat_freq', or 'concat_time'")

    meta = {
        "sfreq": sfreq,
        "ch_names": ch_names,
        "n_mels": int(n_mels),
        "n_fft": int(n_fft),
        "hop_length": int(hop_length),
        "win_length": int(win_length),
        "fmin": float(fmin),
        "fmax": float(fmax),
        "stack_mode": stack_mode,
        "log_base": float(log_base) if log_base != np.e else np.e,
        "center": bool(center),
    }
    return S_out, meta

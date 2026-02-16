import numpy as np
import pandas as pd
import neurokit2 as nk


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


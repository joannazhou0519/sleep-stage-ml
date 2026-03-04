"""
Workflow:
HEADBAND
- events: sub-XX_task-Sleep_acq-headband_events.tsv
- skip whole subject if any stage_hum in BAD_STAGES
- load:   sub-XX_task-Sleep_acq-headband_eeg.edf
- pick channels by NAME: HB_1/HB_2
- epoch using events onset/duration
- features: EEG bandpower + EEG spectrogram
- save:
    OUT_ROOT/sub-XX/headband/features.parquet  (or .csv if parquet unavailable)
    OUT_ROOT/sub-XX/headband/spectrogram.npy

PSG
- events: sub-XX_task-Sleep_acq-psg_events.tsv
- skip whole subject if any stage_hum in BAD_STAGES
- load EEG:  sub-XX_task-Sleep_acq-psg_eeg.edf  pick F3 F4 C3 C4 O1 O2
- load PPG:  same EDF, pick PULSE
- epoch EEG using events onset/duration
- features: EEG bandpower + EEG spectrogram + PPG HRV-proxies (per 30s epoch)
- combine into one dataframe aligned to events
- save:
    OUT_ROOT/sub-XX/psg/features.parquet (or .csv)
    OUT_ROOT/sub-XX/psg/spectrogram.npy
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import compute_eeg_bandpower, compute_eeg_spectrogram, compute_ppg_features
from src.io import save_parquet_or_csv, save_npy, list_subjects, load_events, is_valid_events, load_edf, pick_headband, pick_psg_eeg, pick_psg_ppg, make_epochs


RAW_ROOT = "data/raw"
OUT_ROOT = "data/features"
LOG_PATH = Path(OUT_ROOT) / "dataset_cleaning.log"

# Set SUBJECTS = None to process all sub-* under RAW_ROOT.
SUBJECTS = ["sub-100", "sub-101", "sub-102"]

BAD_STAGES = (8, -2)

SPECTROGRAM_PARAMS = dict(
    n_fft=256,
    hop_length=64,
    win_length=None,
    fmin=0.5,
    fmax=30.0,
    log_base=True,
    eps=1e-10,
    center=False,
)


# -------------------
# helpers
# -------------------

def make_logger(log_path, *, overwrite=True):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if overwrite:
        log_path.write_text("", encoding="utf-8")

    def log(msg):
        line = str(msg)
        print(line)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    return log

def log_run_header(log, settings: dict):
    log("=== dataset_cleaning run ===")
    log("timestamp: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    for k, v in settings.items():
        log(f"{k}: {v}")
    log("")

def process_headband(subject, log):
    acq = "headband"
    try:
        ev_hb = load_events(RAW_ROOT, subject, acq)
        if "offset" in ev_hb.columns and (ev_hb["offset"].to_numpy() != 0).any():
            log(f"[SKIP] {subject} {acq}: nonzero offset present in headband_events.tsv")
            return

        ev = load_events(RAW_ROOT, subject, "psg")

        if "offset" in ev.columns and (ev["offset"].to_numpy() != 0).any():
            log(f"[SKIP] {subject} {acq}: nonzero offset present in psg_events.tsv")
            return

        if "stage_hum" not in ev.columns:
            log(f"[SKIP] {subject} {acq}: no stage_hum column in psg_events.tsv")
            return

        if not is_valid_events(ev, BAD_STAGES):
            log(f"[SKIP] {subject} {acq}: bad stage in PSG labels {BAD_STAGES}")
            return

        raw = load_edf(RAW_ROOT, subject, acq, preload=False)
        raw = pick_headband(raw)

        epochs, y, ev_kept = make_epochs(raw, ev)

        df_bp = compute_eeg_bandpower(epochs, reduce_channels="mean", return_array=False).add_prefix("bp_")
        df = pd.concat([
            pd.DataFrame({
                "subject": subject,
                "acq": acq,
                "epoch_idx": np.arange(len(y)),
                "onset": ev_kept["onset"].to_numpy(float),
                "duration": ev_kept["duration"].to_numpy(float),
                "stage_hum": y.astype(int),   # PSG labels
            }),
            df_bp.reset_index(drop=True),
        ], axis=1)

        S = compute_eeg_spectrogram(epochs, **SPECTROGRAM_PARAMS)

        out_dir = Path(OUT_ROOT) / subject / acq
        save_parquet_or_csv(df, out_dir / "features.parquet")
        save_npy(S, out_dir / "spectrogram.npy")

        log(f"[OK] {subject} {acq}: epochs={len(y)} ch={len(raw.ch_names)} -> {out_dir}")

    except FileNotFoundError as e:
        log(f"[SKIP] {subject} {acq}: missing file -> {e}")
    except Exception as e:
        log(f"[SKIP] {subject} {acq}: {type(e).__name__}: {e}")

def process_psg(subject, log):
    acq = "psg"
    try:
        ev = load_events(RAW_ROOT, subject, acq)
        if "offset" in ev.columns and (ev["offset"].to_numpy() != 0).any():
            log(f"[SKIP] {subject} {acq}: nonzero offset present")
            return
        if "stage_hum" not in ev.columns:
            log(f"[SKIP] {subject} {acq}: no stage_hum column in events.tsv")
            return
        if not is_valid_events(ev, BAD_STAGES):
            log(f"[SKIP] {subject} {acq}: bad stage in {BAD_STAGES}")
            return

        # EEG
        raw_all = load_edf(RAW_ROOT, subject, acq, preload=True)  # preload once
        raw_eeg = pick_psg_eeg(raw_all)
        epochs, y, ev_kept = make_epochs(raw_eeg, ev)

        df_bp = compute_eeg_bandpower(epochs, reduce_channels="mean", return_array=False).add_prefix("eeg_bp_")
        S = compute_eeg_spectrogram(epochs, **SPECTROGRAM_PARAMS)

        # PPG
        raw_ppg = pick_psg_ppg(raw_all)
        ppg = raw_ppg.get_data()[0]
        sfreq_ppg = float(raw_ppg.info["sfreq"])
        epoch_len = float(ev["duration"].iloc[0])
        df_ppg = compute_ppg_features(ppg, sampling_rate=sfreq_ppg, epoch_length=epoch_len)

        # Alignment
        n = len(y)
        df_ppg = df_ppg[df_ppg["epoch"].astype(int) < n].set_index("epoch").reindex(range(n)).reset_index(drop=True)
        df_ppg = df_ppg.add_prefix("ppg_")

        base = pd.DataFrame({
            "subject": subject,
            "acq": acq,
            "epoch_idx": np.arange(n),
            "onset": ev_kept["onset"].to_numpy(float),
            "duration": ev_kept["duration"].to_numpy(float),
            "stage_hum": y.astype(int),
        })

        df = pd.concat([base, df_bp.reset_index(drop=True), df_ppg.reset_index(drop=True)], axis=1)

        out_dir = Path(OUT_ROOT) / subject / acq
        save_parquet_or_csv(df, out_dir / "features.parquet")
        save_npy(S, out_dir / "spectrogram.npy")

        log(f"[OK] {subject} {acq}: epochs={n} eeg_ch={len(raw_eeg.ch_names)} ppg_ch={len(raw_ppg.ch_names)} -> {out_dir}")

    except FileNotFoundError as e:
        log(f"[SKIP] {subject} {acq}: missing file -> {e}")
    except Exception as e:
        log(f"[SKIP] {subject} {acq}: {type(e).__name__}: {e}")


def main():
    if SUBJECTS is None:
        subjects = list_subjects(RAW_ROOT)
    else:
        subjects = SUBJECTS

    log = make_logger(LOG_PATH, overwrite=True)

    log_run_header(log, {
        "RAW_ROOT": RAW_ROOT,
        "OUT_ROOT": OUT_ROOT,
        "SUBJECTS": SUBJECTS if SUBJECTS is not None else "(all sub-*)",
        "BAD_STAGES": BAD_STAGES,
        "SPECTROGRAM_PARAMS": SPECTROGRAM_PARAMS,
    })

    for subject in subjects:
        process_headband(subject, log)
        process_psg(subject, log)

    log("\n=== DONE ===")

if __name__ == "__main__":
    main()

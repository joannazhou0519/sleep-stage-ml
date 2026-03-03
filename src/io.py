import numpy as np
import pandas as pd
import mne
from pathlib import Path

def list_subjects(raw_root):
    raw_root = Path(raw_root)
    subs = [p.name for p in raw_root.glob("sub-*") if p.is_dir()]
    subs.sort(key=lambda s: int("".join([c for c in s if c.isdigit()]) or "999999"))
    return subs

def load_events(raw_root, subject, acq):
    p = Path(raw_root) / subject / "eeg" / f"{subject}_task-Sleep_acq-{acq}_events.tsv"
    df = pd.read_csv(p, sep="\t")

    # normalize column names (handles whitespace and BOM)
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    if df.columns[0].lower().startswith("unnamed"):
        df = df.drop(columns=[df.columns[0]])

    keep = ["onset", "duration"]
    for c in ["offset", "stage_hum", "stage_ai"]:
        if c in df.columns:
            keep.append(c)

    df = df[keep].sort_values("onset").reset_index(drop=True)
    return df

def is_valid_events(df, bad_stages):
    if "offset" in df.columns:
        if (df["offset"].to_numpy() != 0).any():
            return False

    if "stage_hum" not in df.columns:
        return False

    return not df["stage_hum"].isin(set(bad_stages)).any()

def load_edf(raw_root, subject, acq, preload=False):
    p = Path(raw_root) / subject / "eeg" / f"{subject}_task-Sleep_acq-{acq}_eeg.edf"
    return mne.io.read_raw_edf(str(p), preload=preload, verbose="ERROR")

def pick_headband(raw):
    pref = ["HB_1", "HB_2"]
    keep = pref
    if not keep:
        raise ValueError(f"Headband channels not found. Available: {raw.ch_names}")
    raw2 = raw.copy().pick(keep)
    raw2.reorder_channels(keep)
    return raw2

def pick_psg_eeg(raw):
    keep = [ch for ch in ["PSG_F3","PSG_F4","PSG_C3","PSG_C4","PSG_O1","PSG_O2"] if ch in raw.ch_names]
    if not keep:
        raise ValueError(f"PSG EEG channels not found. Available: {raw.ch_names}")
    raw2 = raw.copy().pick(keep)
    raw2.reorder_channels(keep)
    return raw2

def pick_psg_ppg(raw):
    if "PULSE" in raw.ch_names:
        keep = ["PULSE"]
    if not keep:
        raise ValueError(f"PPG channel not found. Available: {raw.ch_names}")
    raw2 = raw.copy().pick(keep)
    raw2.reorder_channels(keep)
    return raw2

def make_epochs(raw, events_df):
    onsets = events_df["onset"].to_numpy(float)
    durations = events_df["duration"].to_numpy(float)
    y = events_df["stage_hum"].to_numpy(int)

    if len(onsets) == 0:
        raise ValueError("events_df is empty")

    dur0 = float(durations[0])
    if not np.allclose(durations, dur0):
        raise ValueError(f"Durations not constant. Unique={np.unique(durations)[:10]}")

    samples = raw.time_as_index(onsets, use_rounding=True)
    events = np.c_[samples, np.zeros(len(samples), int), np.ones(len(samples), int)]

    epochs = mne.Epochs(
        raw, events,
        event_id={"sleep_epoch": 1},
        tmin=0.0, tmax=dur0,
        baseline=None,
        preload=False,
        reject_by_annotation=False,
        verbose="ERROR",
    )

    # keep alignment if any epochs dropped
    keep = np.array([len(r) == 0 for r in epochs.drop_log], dtype=bool)
    events_kept = events_df.loc[keep].reset_index(drop=True)
    y_kept = y[keep]
    return epochs, y_kept, events_kept

def save_parquet_or_csv(df, path_parquet):
    path_parquet = Path(path_parquet)
    path_parquet.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path_parquet, index=False)
    except Exception:
        df.to_csv(path_parquet.with_suffix(".csv"), index=False)

def save_npy(arr, path_npy):
    path_npy = Path(path_npy)
    path_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(path_npy, arr)
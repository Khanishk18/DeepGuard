import os
import librosa
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Input (raw audio) and output (mel-spectrogram) folders
RAW_DIR = "data/audio/raw"
OUT_DIR = "data/audio/mels"

# Audio and mel settings
SR = 16000          # sample rate
N_MELS = 128        # number of Mel bands
HOP = 256           # hop length
DUR = 3.0           # seconds per clip
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg"}

def wav_to_logmel(path):
    """Convert one audio file to normalized log-mel spectrogram."""
    y, sr = librosa.load(path, sr=SR, mono=True)
    # trim silence from start and end
    y, _ = librosa.effects.trim(y, top_db=30)
    target = int(SR * DUR)
    if len(y) < SR // 2:
        raise ValueError("too short (<0.5s)")
    # pad or cut to fixed duration
    y = y[:target] if len(y) >= target else np.pad(y, (0, target - len(y)))
    # create mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP)
    S_db = librosa.power_to_db(S + 1e-9)
    # normalize between 0 and 1
    S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-9)
    return S_db.astype("float32")

def process_class(cls):
    """Process all audio files in one class (real or fake)."""
    in_dir = Path(RAW_DIR) / cls
    out_dir = Path(OUT_DIR) / cls
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [p for p in in_dir.iterdir() if p.suffix.lower() in AUDIO_EXT and p.is_file()]
    if not files:
        print(f"[WARN] No audio files found in {in_dir}")
    ok, bad = 0, 0
    for p in tqdm(files, desc=f"Processing {cls}"):
        try:
            mel = wav_to_logmel(str(p))
            np.save(out_dir / (p.stem + ".npy"), mel)
            ok += 1
        except Exception as e:
            print(f"[SKIP] {p.name}: {e}")
            bad += 1
    print(f"[{cls}] saved: {ok}, skipped: {bad}, out -> {out_dir}")

if __name__ == "__main__":
    process_class("real")
    process_class("fake")
    print("✅ Done! Mel-spectrograms saved in:", OUT_DIR)

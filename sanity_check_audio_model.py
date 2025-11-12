import os, glob, numpy as np, librosa, tensorflow as tf

SR=16000; N_MELS=128; HOP=256; DUR=3.0
MODEL_PATH = "models/audio_model.h5"
RAW_REAL = "data/audio/raw/real"
RAW_FAKE = "data/audio/raw/fake"

def audio_to_logmel(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    target = int(SR*DUR)
    if len(y) < SR//2: 
        return None
    y = y[:target] if len(y)>=target else np.pad(y,(0,target-len(y)))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP)
    S_db = librosa.power_to_db(S + 1e-9)
    S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-9)
    return S_db.astype("float32")[np.newaxis, ..., np.newaxis]

model = tf.keras.models.load_model(MODEL_PATH)

def avg_prob(folder, maxn=10):
    files = sorted(glob.glob(os.path.join(folder, "*")))[:maxn]
    probs = []
    for f in files:
        X = audio_to_logmel(f)
        if X is None: 
            continue
        p = float(model.predict(X, verbose=0)[0][0])
        print(f"{folder} :: {os.path.basename(f)} -> prob={p:.4f}")
        probs.append(p)
    return np.mean(probs) if probs else None

print("\n=== Checking average probabilities (p = sigmoid output) ===")
m_real = avg_prob(RAW_REAL)
m_fake = avg_prob(RAW_FAKE)

print("\nAverage p(real files):", m_real)
print("Average p(fake files):", m_fake)
print("\nInterpretation:")
print("- If p is ~probability of FAKE => expect m_fake > m_real")
print("- If p is ~probability of REAL => expect m_real > m_fake")

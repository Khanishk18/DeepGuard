import os, io
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from PIL import Image
import tensorflow as tf
import librosa

# ------------------ Configuration ------------------
APP_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(APP_DIR, "models")
IMG_SIZE = (160, 160)
SR = 16000
N_MELS = 128
HOP = 256
DUR = 3.0

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "deepguard"  # for flash messages
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB uploads

# ------------------ Load Models ------------------
image_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "image_model.h5")) \
               if os.path.exists(os.path.join(MODELS_DIR, "image_model.h5")) else None
audio_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "audio_model.h5")) \
               if os.path.exists(os.path.join(MODELS_DIR, "audio_model.h5")) else None

# ------------------ Helpers ------------------
def preprocess_image(pil_img):
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    x = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(x, 0)

def audio_to_logmel(file_like):
    y, sr = librosa.load(file_like, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    target = int(SR * DUR)
    if len(y) < SR // 2:  # <0.5s
        raise ValueError("Audio too short")
    y = y[:target] if len(y) >= target else np.pad(y, (0, target - len(y)))
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP)
    S_db = librosa.power_to_db(S + 1e-9)
    S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-9)
    return S_db.astype("float32")[np.newaxis, ..., np.newaxis]  # [1,H,W,1]

# ------------------ Routes ------------------
@app.route("/")
def home():
    return render_template("index.html", has_img=bool(image_model), has_audio=bool(audio_model))

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/threats")
def threats():
    return render_template("threats.html")

@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    if request.method == "POST":
        name = request.form.get("name","").strip()
        email = request.form.get("email","").strip()
        fb = request.form.get("feedback","").strip()
        rating = request.form.get("rating","")
        # You could persist this to a file/db; for demo just flash.
        flash("Thanks for your feedback! 🌟", "ok")
        return redirect(url_for("feedback"))
    return render_template("feedback.html")

# --- Analyze (Image) ---
@app.route("/analyze_image", methods=["POST"])
def analyze_image():
    result = None
    msg = None
    f = request.files.get("image_file")
    if not f or f.filename == "":
        msg = "Please drop an image."
        return render_template("index.html", msg=msg, result=result,
                               has_img=bool(image_model), has_audio=bool(audio_model))
    if not image_model:
        msg = "Image model not available."
        return render_template("index.html", msg=msg, result=result,
                               has_img=bool(image_model), has_audio=bool(audio_model))

    name = f.filename.lower()
    if not any(name.endswith(ext) for ext in [".jpg",".jpeg",".png",".webp"]):
        msg = "Unsupported image type."
    else:
        try:
            buf = io.BytesIO(f.read())
            img = Image.open(buf)
            x = preprocess_image(img)
            p = float(image_model.predict(x, verbose=0)[0][0])
            # flow_from_directory => {'fake':0,'real':1} -> p = P(real)
            label = "Real" if p >= 0.5 else "Fake"
            conf = int(100 * (p if label == "Real" else 1 - p))
            result = {"mode":"Image","label":label,"confidence":conf}
        except Exception as e:
            msg = f"Image error: {e}"
    return render_template("index.html", msg=msg, result=result,
                           has_img=bool(image_model), has_audio=bool(audio_model))

# --- Analyze (Audio) ---
@app.route("/analyze_audio", methods=["POST"])
def analyze_audio():
    result = None
    msg = None
    f = request.files.get("audio_file")
    if not f or f.filename == "":
        msg = "Please drop an audio file."
        return render_template("index.html", msg=msg, result=result,
                               has_img=bool(image_model), has_audio=bool(audio_model))
    if not audio_model:
        msg = "Audio model not available."
        return render_template("index.html", msg=msg, result=result,
                               has_img=bool(image_model), has_audio=bool(audio_model))

    name = f.filename.lower()
    if not any(name.endswith(ext) for ext in [".wav",".mp3",".m4a",".ogg"]):
        msg = "Unsupported audio type."
    else:
        try:
            buf = io.BytesIO(f.read())
            buf.seek(0)
            X = audio_to_logmel(buf)
            p = float(audio_model.predict(X, verbose=0)[0][0])
            print(f"DEBUG audio prob p={p:.4f}")
            AUDIO_THRESHOLD = 0.484  # tuned from your sanity check
            label = "Fake" if p >= AUDIO_THRESHOLD else "Real"
            conf = int(100 * (p if label == "Fake" else 1 - p))
            result = {"mode":"Audio","label":label,"confidence":conf}
        except Exception as e:
            msg = f"Audio error: {e}"

    return render_template("index.html", msg=msg, result=result,
                           has_img=bool(image_model), has_audio=bool(audio_model))

if __name__ == "__main__":
    app.run(debug=True)

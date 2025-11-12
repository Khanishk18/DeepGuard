import os
import numpy as np
import tensorflow as tf
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# -------- settings --------
MEL_DIR = "data/audio/mels"   # where .npy files are saved by prep_audio_to_mels.py
MODEL_OUT = "models/audio_model.h5"
BATCH = 32
EPOCHS = 10
SEED = 42
# --------------------------

def load_mel_arrays(mel_dir):
    """Load mel-spectrogram .npy files from real/ and fake/ into X, y."""
    X, y = [], []
    for label, cls in enumerate(["real", "fake"]):
        cls_dir = os.path.join(mel_dir, cls)
        files = sorted(glob(os.path.join(cls_dir, "*.npy")))
        for f in files:
            arr = np.load(f)             # shape [n_mels, T]
            X.append(arr)
            y.append(label)
    X = np.array(X, dtype="float32")[..., np.newaxis]  # [N, H, W, 1]
    y = np.array(y, dtype="float32")
    return X, y

def make_ds(X, y, train):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if train:
        ds = ds.shuffle(buffer_size=min(len(X), 4096), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return ds

def build_model(input_shape):
    inp = tf.keras.Input(shape=input_shape)  # (H, W, 1)
    x = tf.keras.layers.Conv2D(32, (3,3), padding="same", activation="relu")(inp)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model

def main():
    os.makedirs("models", exist_ok=True)

    # 1) load data
    X, y = load_mel_arrays(MEL_DIR)
    if len(X) == 0:
        raise SystemExit(f"No mel files found under {MEL_DIR}. Run prep_audio_to_mels.py first.")

    # 2) split train/val/test
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp
    )

    # 3) build datasets
    ds_train = make_ds(X_train, y_train, train=True)
    ds_val   = make_ds(X_val,   y_val,   train=False)
    ds_test  = make_ds(X_test,  y_test,  train=False)

    # 4) model
    model = build_model(input_shape=X.shape[1:])

    # 5) callbacks
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor="val_accuracy"),
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy"),
    ]

    # 6) train
    history = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS, callbacks=cbs)

    # 7) evaluate on test
    print("\nEvaluating on test set…")
    test_metrics = model.evaluate(ds_test, verbose=0)
    metric_names = model.metrics_names
    print("Test metrics:")
    for name, val in zip(metric_names, test_metrics):
        print(f"  {name}: {val:.4f}")

    # 8) detailed report
    y_prob = model.predict(ds_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = np.concatenate([y.numpy() for _, y in ds_test]).astype(int)

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=3))

    # 9) save final just in case
    model.save(MODEL_OUT)
    print(f"\n✅ Saved best model to: {MODEL_OUT}")

if __name__ == "__main__":
    main()

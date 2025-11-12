import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# --- Settings ---
IMG_DIR = "data/images"
MODEL_OUT = "models/image_model.h5"
IMG_SIZE = (160, 160)
BATCH = 32
EPOCHS = 10

# --- Data Generators ---
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(IMG_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='binary'
)
val_gen = val_datagen.flow_from_directory(
    os.path.join(IMG_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='binary'
)
test_gen = test_datagen.flow_from_directory(
    os.path.join(IMG_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='binary',
    shuffle=False
)

# --- Model Definition ---
base_model = tf.keras.applications.MobileNetV2(
    weights='imagenet', include_top=False, input_shape=(160, 160, 3)
)
base_model.trainable = False  # freeze base model

x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# --- Training ---
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor="val_accuracy"),
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ]
)

# --- Evaluation ---
print("\nEvaluating on test set...")
test_metrics = model.evaluate(test_gen, verbose=0)
metric_names = model.metrics_names
print("Test metrics:")
for name, val in zip(metric_names, test_metrics):
    print(f"  {name}: {val:.4f}")

# --- Classification report ---
y_prob = model.predict(test_gen, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)
y_true = test_gen.classes

print("\nConfusion matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nClassification report:")
print(classification_report(y_true, y_pred, digits=3))

# --- Save model ---
model.save(MODEL_OUT)
print(f"\n✅ Image model saved to: {MODEL_OUT}")

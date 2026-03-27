import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
import numpy as np


dataset_path = "dataset/final_dataset"

img_size = (224, 224)
batch_size = 32

train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_ds_raw = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_ds_raw.class_names
num_classes = len(class_names)
print("Classes:", class_names)
print("Num classes:", num_classes)

class_counts = {}
for images, labels in train_ds_raw.unbatch():
    lbl = int(labels.numpy())
    class_counts[lbl] = class_counts.get(lbl, 0) + 1

total_samples = sum(class_counts.values())
n_classes = len(class_counts)

class_weight = {
    cls: total_samples / (n_classes * count)
    for cls, count in class_counts.items()
}

print("\nClass weights:")
for cls, w in class_weight.items():
    print(f"  {class_names[cls]}: {w:.3f}")


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds_raw.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds_raw.prefetch(buffer_size=AUTOTUNE)




data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.25),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
], name="augmentation")


base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False   
inputs = keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)   x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)

print("\n========== PHASE 1: Training classification head ==========")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks_phase1 = [
    keras.callbacks.ModelCheckpoint(
        "best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    class_weight=class_weight,
    callbacks=callbacks_phase1
)


print("\n========== PHASE 2: Fine-tuning MobileNetV2 top layers ==========")


base_model.trainable = True
fine_tune_at = len(base_model.layers) - 30
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

print(f"Fine-tuning from layer {fine_tune_at} / {len(base_model.layers)}")


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_phase2 = [
    keras.callbacks.ModelCheckpoint(
        "best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-8,
        verbose=1
    )
]

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    class_weight=class_weight,
    callbacks=callbacks_phase2
)

val_loss, val_acc = model.evaluate(val_ds)
print(f"Val Loss:     {val_loss:.4f}")
print(f"Val Accuracy: {val_acc*100:.2f}%")
model.save("multi_crop_model.keras")
print("Training finished! Best model saved to best_model.keras")

y_true, y_pred = [], []

for images, labels in val_ds_raw:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

for i, name in enumerate(class_names):
    mask = y_true == i
    if mask.sum() == 0:
        continue
    acc = (y_pred[mask] == i).mean() * 100
    print(f"  {name:<40} {acc:.1f}%  ({mask.sum()} samples)")

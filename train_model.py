import os
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ================= AUTO DETECT DATASET =================
PROJECT_ROOT = r"C:\Users\kumar\OneDrive\Attachments\Desktop\Megha Di Project"

TRAIN_DIR = VALID_DIR = TEST_DIR = None
for root, dirs, files in os.walk(PROJECT_ROOT):
    if os.path.basename(root).lower() == "train":
        TRAIN_DIR = root
    elif os.path.basename(root).lower() == "valid":
        VALID_DIR = root
    elif os.path.basename(root).lower() == "test":
        TEST_DIR = root

if not TRAIN_DIR or not VALID_DIR:
    raise FileNotFoundError("❌ train/valid folder not found")

print("TRAIN:", TRAIN_DIR)
print("VALID:", VALID_DIR)
print("TEST :", TEST_DIR)

# ================= PARAMETERS =================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

# ================= DATA GENERATORS =================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True
)

valid_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

valid_data = valid_gen.flow_from_directory(
    VALID_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# ================= SAVE CLASS ORDER (CRITICAL) =================
class_names = list(train_data.class_indices.keys())
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

print("✅ Class names saved")

# ================= MODEL =================
model = Sequential([
    Conv2D(32, 3, activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(),

    Conv2D(64, 3, activation="relu"),
    MaxPooling2D(),

    Conv2D(128, 3, activation="relu"),
    MaxPooling2D(),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_data, validation_data=valid_data, epochs=EPOCHS)

model.save("plant_disease_model.h5")
print("✅ Model saved")
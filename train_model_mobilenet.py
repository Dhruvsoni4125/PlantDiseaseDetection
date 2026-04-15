import os
import json
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==================================================
# AUTO-DETECT DATASET
# ==================================================
PROJECT_ROOT = r"C:\Users\kumar\OneDrive\Attachments\Desktop\Megha Di Project"

TRAIN_DIR = VALID_DIR = None
for root, dirs, files in os.walk(PROJECT_ROOT):
    if os.path.basename(root).lower() == "train":
        TRAIN_DIR = root
    elif os.path.basename(root).lower() == "valid":
        VALID_DIR = root

if not TRAIN_DIR or not VALID_DIR:
    raise FileNotFoundError("❌ train / valid folder not found")

print("TRAIN:", TRAIN_DIR)
print("VALID:", VALID_DIR)

# ==================================================
# PARAMETERS
# ==================================================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 6   # 👈 MUCH LESS THAN BEFORE

# ==================================================
# DATA GENERATORS
# ==================================================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
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

# ==================================================
# SAVE CLASS NAMES (CRITICAL)
# ==================================================
class_names = list(train_data.class_indices.keys())
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

print("✅ class_names.json saved")

# ==================================================
# MOBILENET MODEL
# ==================================================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False  # 🔥 KEY FOR FAST TRAINING

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(len(class_names), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==================================================
# TRAIN
# ==================================================
model.fit(
    train_data,
    validation_data=valid_data,
    epochs=EPOCHS
)

# ==================================================
# SAVE MODEL 
# ==================================================
model.save("plant_disease_model.h5")
print("✅ MobileNet model saved")
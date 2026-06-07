import tensorflow as tf
from tensorflow.keras import layers, models
import os

# 📁 Dataset path
dataset_path = r"C:\xampp\htdocs\test\backend\datasets\train"

# 🏷️ Classes (must match folder names)
classes = ["Joy", "Sadness", "Calmness", "Anger"]

# 📦 Load dataset
img_size = (224, 224)
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="categorical"
)

# ⚡ Normalize images
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# 🧠 Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(4, activation='softmax')
])
#
# ⚙️ Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 🚀 Train model
model.fit(train_ds, epochs=10)

# 💾 Save model

model.save("model.keras")

print("✅ Model training complete and saved as model.h5")
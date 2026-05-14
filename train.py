import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# ✅ Corrected Dataset Path
DATASET_PATH = "/kaggle/input/3-kinds-of-pneumonia/Curated X-Ray Dataset"

# ✅ Image Settings
IMG_SIZE = (150, 150)
BATCH_SIZE = 16  

# ✅ Data Augmentation & Rescaling
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# ✅ Load Training Data
train_data = datagen.flow_from_directory(
    DATASET_PATH, 
    target_size=IMG_SIZE, 
    batch_size=BATCH_SIZE, 
    class_mode='categorical',  # ✅ Ensure one-hot encoding
    subset='training',
    shuffle=True
)

# ✅ Load Validation Data
val_data = datagen.flow_from_directory(
    DATASET_PATH, 
    target_size=IMG_SIZE, 
    batch_size=BATCH_SIZE, 
    class_mode='categorical',  # ✅ Ensure one-hot encoding
    subset='validation',
    shuffle=False
)

# ✅ Check Class Labels
print("Class Labels:", train_data.class_indices)

# ✅ Define CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # ✅ Ensure 4 output neurons for 4 classes
])

# ✅ Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Train the Model
print("🔄 Training started...")
history = model.fit(train_data, validation_data=val_data, epochs=10)
print("✅ Training completed successfully!")

# ✅ Save the Model
model.save('/kaggle/working/pneumonia_model.h5')
print("✅ Model saved at '/kaggle/working/pneumonia_model.h5'")

# ✅ Evaluate the Model
val_data.reset()
y_pred = model.predict(val_data)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert to class labels
y_true = val_data.classes  

# ✅ Classification Report
print("📊 Classification Report:\n", classification_report(y_true, y_pred_classes, target_names=list(train_data.class_indices.keys())))

# ✅ Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=train_data.class_indices.keys(), yticklabels=train_data.class_indices.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.show()

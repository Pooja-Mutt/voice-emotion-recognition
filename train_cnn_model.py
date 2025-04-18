import os
import numpy as np
import librosa
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

DATA_DIR = "data"
SAMPLE_RATE = 22050
N_MFCC = 40
MAX_LEN = 130  # ~3 seconds of audio

# === Step 1: Extract 2D MFCCs ===
X = []
y = []

print("🎧 Extracting 2D MFCCs for CNN...")
for emotion in os.listdir(DATA_DIR):
    emotion_path = os.path.join(DATA_DIR, emotion)
    if os.path.isdir(emotion_path):
        for file in os.listdir(emotion_path):
            if file.endswith(".wav"):
                file_path = os.path.join(emotion_path, file)
                audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)

                # Pad MFCC to fixed length
                if mfcc.shape[1] < MAX_LEN:
                    pad_width = MAX_LEN - mfcc.shape[1]
                    mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
                else:
                    mfcc = mfcc[:, :MAX_LEN]

                X.append(mfcc)
                y.append(emotion)

X = np.array(X)
y = np.array(y)

# Reshape for CNN: (samples, height, width, channels)
X = X[..., np.newaxis]  # Add channel dimension

print(f"✅ X shape: {X.shape}, y shape: {y.shape}")

# === Step 2: Encode labels ===
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Save label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# === Step 3: Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, stratify=y_encoded, random_state=42
)

# === Step 4: Build CNN Model ===
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(N_MFCC, MAX_LEN, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(y_categorical.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === Step 5: Train ===
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=8,
    callbacks=[early_stop],
    verbose=1
)

model.save("model.h5")
print("✅ CNN model trained and saved!")

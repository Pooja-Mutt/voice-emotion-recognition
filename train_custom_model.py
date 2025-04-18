import os
import numpy as np
import librosa
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

DATA_DIR = "data"
SAMPLE_RATE = 22050
N_MFCC = 40

# === Step 1: Load all audio files and extract MFCC features ===
X = []
y = []

print("🎧 Extracting features from your custom recordings...")
for emotion in os.listdir(DATA_DIR):
    emotion_path = os.path.join(DATA_DIR, emotion)
    if os.path.isdir(emotion_path):
        for file in os.listdir(emotion_path):
            if file.endswith(".wav"):
                file_path = os.path.join(emotion_path, file)
                audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                if len(audio) < sr:
                    audio = np.pad(audio, (0, sr - len(audio)))
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
                mfcc_mean = np.mean(mfcc.T, axis=0)
                X.append(mfcc_mean)
                y.append(emotion)

X = np.array(X)
y = np.array(y)

print(f"✅ Loaded {len(X)} samples across {len(set(y))} emotions")

# === Step 2: Label encode and scale ===
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save encoder and scaler
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# === Step 3: Train/Test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

# === Step 4: Build and Train Model ===
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(N_MFCC,)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(y_categorical.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=8,
    callbacks=[early_stop],
    verbose=1
)

model.save("model.h5")
print("✅ Custom model trained and saved!")

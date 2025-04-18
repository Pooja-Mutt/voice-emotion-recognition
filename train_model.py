import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

from label_ravdess import read_voice_files
from extract_features import extract_features

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# === Step 1: Load and Filter Data ===
folder_path = r"C:\Users\Pooja Mutt\Downloads\archive"
df = read_voice_files(folder_path)

features = []
labels = []

print("🔍 Extracting features...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    mfcc = extract_features(row['file'])
    if mfcc is not None:
        features.append(mfcc)
        labels.append(row['emotion'])

X = np.array(features)
y = np.array(labels)

print("✅ Features shape:", X.shape)
print("✅ Labels shape:", y.shape)

# === Step 2: Encode labels and normalize features ===
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save label encoder and scaler
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# === Step 3: Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

# === Step 4: Build Model ===
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(40,)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(y_categorical.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# === Step 5: Train Model ===
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop]
)

# === Step 6: Save model ===
model.save("model.h5")
print("✅ Model saved as model.h5")


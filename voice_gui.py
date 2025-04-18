import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tkinter as tk
import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
import tensorflow as tf
import pickle

SAMPLE_RATE = 22050
DURATION = 3
N_MFCC = 40
MAX_LEN = 130
WAV_FILENAME = "live_input.wav"

# Load model and label encoder
model = tf.keras.models.load_model("model.h5")
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Extract features as 2D MFCCs for CNN
def extract_features(file_path):
    import soundfile as sf
    import numpy as np
    import librosa

    # Load audio freshly every time
    audio, sr = sf.read(file_path, always_2d=False)

    # Convert stereo to mono if needed
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Normalize
    if np.max(np.abs(audio)) != 0:
        audio = audio / np.max(np.abs(audio))

    # Ensure at least DURATION seconds of audio
    required_len = sr * DURATION
    if len(audio) < required_len:
        audio = np.pad(audio, (0, required_len - len(audio)))

    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)

    # Pad/crop to MAX_LEN frames
    if mfcc.shape[1] < MAX_LEN:
        mfcc = np.pad(mfcc, ((0, 0), (0, MAX_LEN - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_LEN]

    # Final shape: (1, 40, 130, 1)
    return mfcc.reshape(1, N_MFCC, MAX_LEN, 1)

# Record voice for prediction
def record_voice():
    print("🎙️ Recording...")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    sf.write(WAV_FILENAME, recording, SAMPLE_RATE)
    print("✅ Recording saved")

# Predict emotion
def predict_emotion():
    record_voice()
    features = extract_features(WAV_FILENAME)
    prediction = model.predict(features)[0]

    emotion_labels = encoder.inverse_transform(np.arange(len(prediction)))
    sorted_indices = np.argsort(prediction)[::-1]

    result_text = "Emotion Probabilities:\n"
    for idx in sorted_indices:
        label = emotion_labels[idx]
        confidence = prediction[idx] * 100
        result_text += f"{label}: {confidence:.1f}%\n"

    result_label.config(text=result_text.strip(), fg="blue")

# GUI setup
app = tk.Tk()
app.title("🎤 CNN Voice Emotion Detector")

record_btn = tk.Button(app, text="🎙️ Record and Predict", command=predict_emotion, font=("Arial", 14))
record_btn.pack(pady=10)

result_label = tk.Label(app, text="Speak and see your emotion!", font=("Arial", 14), justify="left", wraplength=300)
result_label.pack(pady=20)

app.mainloop()

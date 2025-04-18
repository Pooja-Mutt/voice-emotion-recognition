import os
import librosa
import soundfile as sf
import numpy as np

AUGMENTATIONS_PER_FILE = 2  # Number of augmented versions per original
SAMPLE_RATE = 22050
BASE_DIR = "data"

# === Augmentation Functions ===

def pitch_shift(audio, sr):
    # ✅ Use keyword arguments to avoid TypeError in newer Librosa
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=np.random.uniform(-2, 2))

def stretch_audio(audio, sr):
    stretch_rate = np.random.uniform(0.8, 1.2)
    return librosa.effects.time_stretch(y=audio, rate=stretch_rate)


def add_volume(audio):
    gain = np.random.uniform(0.7, 1.3)
    return audio * gain

def augment_audio(audio, sr):
    aug_audio = audio
    aug_type = np.random.choice(["pitch", "stretch", "volume"])

    if aug_type == "pitch":
        aug_audio = pitch_shift(audio, sr)
    elif aug_type == "stretch":
        aug_audio = stretch_audio(audio, sr)
    elif aug_type == "volume":
        aug_audio = add_volume(audio)

    return aug_audio

# === Process Folder ===

def process_folder(emotion_folder):
    for filename in os.listdir(emotion_folder):
        if filename.endswith(".wav") and not filename.startswith("aug_"):
            path = os.path.join(emotion_folder, filename)
            audio, sr = librosa.load(path, sr=SAMPLE_RATE)

            for i in range(AUGMENTATIONS_PER_FILE):
                aug_audio = augment_audio(audio, sr)

                # Match original length
                if len(aug_audio) > len(audio):
                    aug_audio = aug_audio[:len(audio)]
                elif len(aug_audio) < len(audio):
                    aug_audio = np.pad(aug_audio, (0, len(audio) - len(aug_audio)))

                aug_filename = os.path.join(
                    emotion_folder,
                    f"aug_{i}_{filename}"
                )
                sf.write(aug_filename, aug_audio, sr)
                print(f"✅ Augmented saved: {aug_filename}")

# === Main Entry Point ===

def main():
    print("🔄 Starting audio augmentation...")
    for emotion in os.listdir(BASE_DIR):
        emotion_path = os.path.join(BASE_DIR, emotion)
        if os.path.isdir(emotion_path):
            process_folder(emotion_path)
    print("✅ All augmentations complete!")

if __name__ == "__main__":
    main()

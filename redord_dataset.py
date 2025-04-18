import os
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 22050
DURATION = 3  # seconds per recording
BASE_DIR = "data"

emotions = ["happy", "sad", "angry", "neutral"]

def record_sample(emotion, index):
    print(f"\n🎤 Say something {emotion.upper()} (Recording {index})... 🎬")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    folder = os.path.join(BASE_DIR, emotion)
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"{emotion}_{index}.wav")
    sf.write(filename, recording, SAMPLE_RATE)
    print(f"✅ Saved: {filename}")

def main():
    print("🎙️ Starting voice emotion dataset recorder")
    print(f"Each recording will be {DURATION} seconds long")

    for emotion in emotions:
        count = int(input(f"\nHow many samples do you want to record for '{emotion}'? "))
        for i in range(1, count + 1):
            input(f"Press Enter to record sample {i} for '{emotion}'...")
            record_sample(emotion, i)

    print("\n✅ Recording complete! All samples saved in 'data/' folder.")

if __name__ == "__main__":
    main()

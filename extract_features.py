import librosa
import numpy as np
import soundfile as sf

def extract_features(file_path):
    try:
        # Load the audio file using soundfile
        audio, sample_rate = sf.read(file_path)

        # If stereo, convert to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            audio = audio / np.max(np.abs(audio))  # Normalize to [-1, 1]

        # Extract MFCCs using librosa
        if len(audio) < sample_rate:
            padding = sample_rate - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')

        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)

        return mfccs_processed

    except Exception as e:
        print("Error processing:", file_path)
        return None

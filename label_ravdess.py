   #print(df['emotion'].value_counts())  # show how many of each emotion
import os
import pandas as pd

# This dictionary maps emotion codes to actual emotion names
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Function to scan folders and create a table with file path and emotion
def read_voice_files(folder):
    data = []
    for actor_folder in os.listdir(folder):
        actor_path = os.path.join(folder, actor_folder)
        if os.path.isdir(actor_path):
            for file in os.listdir(actor_path):
                if file.endswith(".wav"):
                    emotion_code = file.split("-")[2]
                    emotion = emotion_map.get(emotion_code)
                    full_path = os.path.join(actor_path, file)
                    data.append([full_path, emotion])
    return pd.DataFrame(data, columns=["file", "emotion"])

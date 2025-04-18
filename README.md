# 🎙️ Voice Emotion Recognition (RAVDESS Dataset)

This project uses a custom-trained CNN model to detect **emotions from voice recordings**. It features a real-time Tkinter GUI that records input and classifies the emotion using speech features like MFCCs.

---

## 📸 Demo
> 🎤 Say something like *“I’m very happy today!”* and watch the app predict emotions live.

![App Demo](screenshot.png) <!-- Add a screenshot if available -->

---

## 🔧 Features

- 🎛️ Tkinter-based Voice Recording GUI
- 📊 Real-time emotion classification
- 🤖 Custom-trained CNN model using RAVDESS
- 📁 Augmented audio training with `librosa`
- 📦 Exported `.h5` and `.pkl` model files for reuse

---

## 🧠 Emotion Categories

- Neutral  
- Calm  
- Happy  
- Sad  
- Angry  
- Fearful  
- Disgust  
- Surprised  

---

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/Pooja-Mutt/voice-emotion-recognition.git
   cd voice-emotion-recognition

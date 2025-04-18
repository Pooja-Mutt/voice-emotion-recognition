from label_ravdess import read_voice_files
from extract_features import extract_features
import numpy as np
from tqdm import tqdm
import pickle

# Step 1: Load data
folder_path = r"C:\Users\Pooja Mutt\Downloads\archive"
df = read_voice_files(folder_path)

features = []
labels = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    mfccs = extract_features(row['file'])
    if mfccs is not None:
        features.append(mfccs)
        labels.append(row['emotion'])

X = np.array(features)
y = np.array(labels)

print("Feature shape:", X.shape)
print("Label shape:", y.shape)

# Step 2: Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Save the label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, shuffle=True, stratify=y_encoded
)


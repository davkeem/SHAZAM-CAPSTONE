import librosa
import numpy as np

# Load an example audio file or replace with your own
y, sr = librosa.load('your_audio_file.wav')

features = {}

# 1. MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
features['mfccs_mean'] = np.mean(mfccs, axis=1)
features['mfccs_std'] = np.std(mfccs, axis=1)

# 2. Chroma Features
chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
features['chroma_mean'] = np.mean(chroma_stft, axis=1)
features['chroma_std'] = np.std(chroma_stft, axis=1)

# 3. Spectral Contrast
spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
features['spectral_contrast_mean'] = np.mean(spectral_contrast, axis=1)
features['spectral_contrast_std'] = np.std(spectral_contrast, axis=1)

# 4. Spectral Centroid
spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
features['spectral_centroid_mean'] = np.mean(spectral_centroid)
features['spectral_centroid_std'] = np.std(spectral_centroid)

# 5. Spectral Bandwidth
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)

# 6. Spectral Rolloff
spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
features['spectral_rolloff_std'] = np.std(spectral_rolloff)

# 7. Tempo
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
features['tempo'] = tempo

# 8. Onset Detection
onset_env = librosa.onset.onset_strength(y=y, sr=sr)
features['onset_env_mean'] = np.mean(onset_env)
features['onset_env_std'] = np.std(onset_env)

# 9. Beat Tracking
_, beats = librosa.beat.beat_track(y=y, sr=sr)
features['beats'] = len(beats)

# 10. Harmonic/Percussive Source Separation
harmonic = librosa.effects.harmonic(y)
percussive = librosa.effects.percussive(y)

features['harmonic_mean'] = np.mean(harmonic)
features['harmonic_std'] = np.std(harmonic)
features['percussive_mean'] = np.mean(percussive)
features['percussive_std'] = np.std(percussive)

# Output features
print(features)

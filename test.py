import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the audio file
test_path = r"C:\Users\david\Downloads\Wav\01_uptown_funk.wav"
y, sr = librosa.load(test_path, sr=None)

# Extract MFCCs with default hop length
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Get hop length from librosa's default or set it explicitly
hop_length = 512  # default in librosa
frame_duration = hop_length / sr  # Duration of each frame in seconds

# Calculate the number of frames per 0.5 seconds
frames_per_segment = int(0.5 / frame_duration)
print(frames_per_segment)

# Trim to ensure full segments, if necessary
n_segments = mfccs.shape[1] // frames_per_segment
print(mfccs.shape[1])
print(n_segments)
mfccs_trimmed = mfccs[:, :n_segments * frames_per_segment]

# Reshape and average over each segment
mfccs_avg = mfccs_trimmed.reshape(13, n_segments, frames_per_segment).mean(axis=2)

print(mfccs_avg.shape)  # (13, n_segments)
plt.figure(figsize=(10, 6))

librosa.display.specshow(mfccs_avg, sr=sr, x_axis='time', cmap='coolwarm')

plt.title(f'MFCC Heatmap')
plt.ylabel('MFCC Coefficients')
plt.xlabel('Time (s)')
plt.colorbar(format='%+2.0f dB')

plt.tight_layout()
plt.show()

plt.close()
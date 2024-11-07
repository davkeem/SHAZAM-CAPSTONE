import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

test_path = r"C:\Users\david\Downloads\Wav\01_uptown_funk.wav"
y, sr = librosa.load(test_path, sr=None)
res = librosa.resample(y=y, orig_sr=sr, target_sr=22000)
target_hop_length = int(22000 / 22)  # Approximate hop length to get 22 frames per second
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
re_mfccs = librosa.feature.mfcc(y=res, sr=22000, n_mfcc=13, hop_length=target_hop_length)

print(mfccs.shape)
print(re_mfccs.shape)
plt.figure(figsize=(10, 6))

librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap='coolwarm')

plt.title(f'MFCC Heatmap')
plt.ylabel('MFCC Coefficients')
plt.xlabel('Time (s)')
plt.colorbar(format='%+2.0f dB')

plt.tight_layout()
plt.show()

plt.close()
"""
(13, 23234)
(13, 5935)

(13, 11896)
(13, 5935)
"""

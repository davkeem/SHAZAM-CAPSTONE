# Example code to generate chroma graph 

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load the full audio file
# add duration=time to librosa.load to specify time length (e.g. duration=30 for 30 sec, etc.)
y, sr = librosa.load(r'\users\josep\Downloads\Wav\20_fight_song.wav', sr=None)
# Change above file path to your local file path!!!

# Step 2: Set the hop length and extract chroma features
hop_length = 512  # Adjust this if necessary
chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)

# Step 3: Calculate the duration of the audio in seconds
duration = librosa.get_duration(y=y, sr=sr)

# Step 4: Set the correct time axis based on the hop length and duration
frames = np.arange(chroma_stft.shape[1])
time_axis = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

# Step 5: Plot the chroma features with the correct time axis
plt.figure(figsize=(10, 6))

# Display the chroma features
librosa.display.specshow(chroma_stft, y_axis='chroma', x_axis='time', cmap='magma', hop_length=hop_length, sr=sr)

# Add color bar and labels
plt.colorbar(format='%+2.0f')
plt.title('Chroma Feature Heatmap')
plt.tight_layout()
plt.show()

# Example code to generate 13 MFCC graph

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load the full audio file
# CHANGE THIS TO WHEREVER YOUR LOCAL FILE IS!!!
y, sr = librosa.load(r'\users\josep\Downloads\Wav\21_679.wav', sr=None)


# Step 2: Extract 13 MFCCs from the audio file
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Step 3: Create a heatmap to visualize the MFCCs
plt.figure(figsize=(10, 6))  # Set the figure size for better visualization

# Plot the MFCC heatmap using librosa's display function
librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap='coolwarm')

# Add color bar to show the intensity scale
plt.colorbar(format='%+2.0f dB')

# Add title and labels
plt.title('MFCC Heatmap (13 Coefficients)')
plt.ylabel('MFCC Coefficients')
plt.xlabel('Time (s)')

# Display the plot
plt.tight_layout()
plt.show()

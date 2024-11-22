import numpy as np
import pandas as pd
import os
import librosa

# Input and output directories
audio_folder = r"C:\Users\ulalenepi\Downloads\Wav"
audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]
audio_files_sorted = sorted(audio_files, key=lambda x: int(x.split('_')[0]))
audio_paths = [os.path.join(audio_folder, f) for f in audio_files_sorted]

process_folder = r"C:\Users\ulalenepi\OneDrive - University of Alaska\Documents\GitHub\SHAZAM-CAPSTONE\processed_spectral_centroid"

# Ensure output folder exists
os.makedirs(process_folder, exist_ok=True)

# Loop over all audio files
for i, file in enumerate(audio_paths):
    # Load the audio file
    y, sr = librosa.load(file, sr=None)

    # Extract spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    # Get hop length (default in librosa)
    hop_length = 512
    frame_duration = hop_length / sr  # Duration of each frame in seconds

    # Calculate the number of frames per 0.5 seconds
    frames_per_segment = int(0.5 / frame_duration)
    print(f"Frames per segment: {frames_per_segment}")

    # Trim to ensure full segments
    n_segments = spectral_centroid.shape[1] // frames_per_segment
    print(f"Spectral Centroid shape: {spectral_centroid.shape[1]} frames, {n_segments} segments")
    spectral_centroid_trimmed = spectral_centroid[:, :n_segments * frames_per_segment]

    # Reshape and average over each segment
    spectral_centroid_avg = spectral_centroid_trimmed.reshape(1, n_segments, frames_per_segment).mean(axis=2)

    # Save the averaged spectral centroid to a CSV file
    output_filename = os.path.join(process_folder, f"{os.path.splitext(audio_files_sorted[i])[0]}_processed_centroid.csv")
    df = pd.DataFrame(spectral_centroid_avg.T, columns=['Spectral Centroid'])  # Transpose to (n_segments, 1)
    df.to_csv(output_filename, index=False, header=True)

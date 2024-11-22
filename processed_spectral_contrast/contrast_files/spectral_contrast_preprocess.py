import numpy as np
import pandas as pd
import os
import librosa

# Paths to audio files and output folder
audio_folder = r"C:\Users\ulalenepi\Downloads\Wav"
audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]
audio_files_sorted = sorted(audio_files, key=lambda x: int(x.split('_')[0]))
audio_paths = [os.path.join(audio_folder, f) for f in audio_files_sorted]

process_folder = r"C:\Users\ulalenepi\OneDrive - University of Alaska\Documents\GitHub\SHAZAM-CAPSTONE\processed_spectral_contrast"
os.makedirs(process_folder, exist_ok=True)

# Loop over all audio files
for i, file in enumerate(audio_paths):
    # Load the audio file
    y, sr = librosa.load(file, sr=None)

    # Extract spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    # Get hop length from librosa's default or set it explicitly
    hop_length = 512  # default in librosa
    frame_duration = hop_length / sr  # Duration of each frame in seconds

    # Calculate the number of frames per 0.5 seconds
    frames_per_segment = int(0.5 / frame_duration)
    print(f"Frames per segment: {frames_per_segment}")

    # Trim to ensure full segments, if necessary
    n_segments = spectral_contrast.shape[1] // frames_per_segment
    print(f"Spectral Contrast shape: {spectral_contrast.shape[1]} frames, {n_segments} segments")
    spectral_contrast_trimmed = spectral_contrast[:, :n_segments * frames_per_segment]

    # Reshape and average over each segment
    spectral_contrast_avg = spectral_contrast_trimmed.reshape(spectral_contrast.shape[0], n_segments, frames_per_segment).mean(axis=2)

    # Save the averaged spectral contrast to a CSV file
    output_filename = os.path.join(process_folder, f"{os.path.splitext(audio_files_sorted[i])[0]}_processed_contrast.csv")
    df = pd.DataFrame(spectral_contrast_avg.T)  # Transpose to (n_segments, n_bands) before saving
    df.to_csv(output_filename, index=False, header=[f'Band_{i}' for i in range(spectral_contrast.shape[0])])

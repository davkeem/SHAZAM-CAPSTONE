import numpy as np
import pandas as pd
import os
import librosa

# Define input and output folders
audio_folder = r"C:\Users\ulalenepi\Downloads\Wav"
audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]
audio_files_sorted = sorted(audio_files, key=lambda x: int(x.split('_')[0]))
audio_paths = [os.path.join(audio_folder, f) for f in audio_files_sorted]

# Correct paths for processed harmonic and percussive feature outputs
process_folder_harmonic = r"C:\Users\ulalenepi\OneDrive - University of Alaska\Documents\GitHub\SHAZAM-CAPSTONE\dataset_processed\processed_harmonic"
process_folder_percussive = r"C:\Users\ulalenepi\OneDrive - University of Alaska\Documents\GitHub\SHAZAM-CAPSTONE\dataset_processed\processed_percussive"

# Create output directories if they don't exist
os.makedirs(process_folder_harmonic, exist_ok=True)
os.makedirs(process_folder_percussive, exist_ok=True)

# Loop over all audio files
for i, file in enumerate(audio_paths):

    # Load the audio file
    y, sr = librosa.load(file, sr=None)

    # Perform harmonic-percussive source separation
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Extract features for harmonic component
    mfccs_harmonic = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=13)

    # Extract features for percussive component
    mfccs_percussive = librosa.feature.mfcc(y=y_percussive, sr=sr, n_mfcc=13)

    # Get hop length from librosa's default or set it explicitly
    hop_length = 512  # default in librosa
    frame_duration = hop_length / sr  # Duration of each frame in seconds

    # Calculate the number of frames per 0.5 seconds
    frames_per_segment = int(0.5 / frame_duration)

    # Trim and segment for harmonic features
    n_segments_harmonic = mfccs_harmonic.shape[1] // frames_per_segment
    mfccs_harmonic_trimmed = mfccs_harmonic[:, :n_segments_harmonic * frames_per_segment]
    mfccs_harmonic_avg = mfccs_harmonic_trimmed.reshape(13, n_segments_harmonic, frames_per_segment).mean(axis=2)

    # Trim and segment for percussive features
    n_segments_percussive = mfccs_percussive.shape[1] // frames_per_segment
    mfccs_percussive_trimmed = mfccs_percussive[:, :n_segments_percussive * frames_per_segment]
    mfccs_percussive_avg = mfccs_percussive_trimmed.reshape(13, n_segments_percussive, frames_per_segment).mean(axis=2)

    # Save harmonic features to a CSV file
    output_filename_harmonic = os.path.join(
        process_folder_harmonic,
        f"{os.path.splitext(audio_files_sorted[i])[0]}_processed_harmonic.csv"
    )
    df_harmonic = pd.DataFrame(mfccs_harmonic_avg.T)  # Transpose to (n_segments, 13) before saving
    df_harmonic.to_csv(output_filename_harmonic, index=False, header=True)

    # Save percussive features to a CSV file
    output_filename_percussive = os.path.join(
        process_folder_percussive,
        f"{os.path.splitext(audio_files_sorted[i])[0]}_processed_percussive.csv"
    )
    df_percussive = pd.DataFrame(mfccs_percussive_avg.T)  # Transpose to (n_segments, 13) before saving
    df_percussive.to_csv(output_filename_percussive, index=False, header=True)

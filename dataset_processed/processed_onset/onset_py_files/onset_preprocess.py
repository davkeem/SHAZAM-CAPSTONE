import numpy as np
import pandas as pd
import os
import librosa

audio_folder = r"C:\Users\ulalenepi\Downloads\Wav"
audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]
audio_files_sorted = sorted(audio_files, key=lambda x: int(x.split('_')[0]))
audio_paths = [os.path.join(audio_folder, f) for f in audio_files_sorted]

process_folder = r"C:\Users\ulalenepi\OneDrive - University of Alaska\Documents\GitHub\SHAZAM-CAPSTONE\dataset_processed\processed_onset"

# Ensure the process folder exists
os.makedirs(process_folder, exist_ok=True)

# Loop over all audio files
for i, file in enumerate(audio_paths):

    # Load the audio file
    y, sr = librosa.load(file, sr=None)

    # Calculate onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    # Get hop length from librosa's default or set it explicitly
    hop_length = 512  # default in librosa
    frame_duration = hop_length / sr  # Duration of each frame in seconds

    # Calculate the number of frames per 0.5 seconds
    frames_per_segment = int(0.5 / frame_duration)
    print(f"Frames per segment: {frames_per_segment}")

    # Trim to ensure full segments, if necessary
    n_segments = len(onset_env) // frames_per_segment
    print(f"Onset strength shape: {len(onset_env)} frames, {n_segments} segments")
    onset_env_trimmed = onset_env[:n_segments * frames_per_segment]

    # Reshape and average over each segment
    onset_avg = onset_env_trimmed.reshape(n_segments, frames_per_segment).mean(axis=1)

    # Save the averaged onset strengths to a CSV file
    output_filename = os.path.join(process_folder, f"{os.path.splitext(audio_files_sorted[i])[0]}_processed_onset.csv")
    df = pd.DataFrame(onset_avg, columns=['Onset Strength'])  # Single column for averaged onset strengths
    df.to_csv(output_filename, index=False, header=True)

print("Onset detection processing completed.")

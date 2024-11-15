import numpy as np
import pandas as pd
import os
import librosa

audio_folder = r"C:\Users\david\Downloads\Wav"
audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]
audio_files_sorted = sorted(audio_files, key=lambda x: int(x.split('_')[0]))
audio_paths = [os.path.join(audio_folder, f) for f in audio_files_sorted]

process_folder = r"C:\Users\david\fall2024\capstone\SHAZAM-CAPSTONE\librosa_dataset\mfcc\processed_mfcc"

# Loop over all audio files
for i, file in enumerate(audio_paths):

    # Load the audio file
    y, sr = librosa.load(file, sr=None)

    # Extract MFCCs with default hop length
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Get hop length from librosa's default or set it explicitly
    hop_length = 512  # default in librosa
    frame_duration = hop_length / sr  # Duration of each frame in seconds

    # Calculate the number of frames per 0.5 seconds
    frames_per_segment = int(0.5 / frame_duration)
    print(f"Frames per segment: {frames_per_segment}")

    # Trim to ensure full segments, if necessary
    n_segments = mfccs.shape[1] // frames_per_segment
    print(f"MFCCs shape: {mfccs.shape[1]} frames, {n_segments} segments")
    mfccs_trimmed = mfccs[:, :n_segments * frames_per_segment]

    # Reshape and average over each segment
    mfccs_avg = mfccs_trimmed.reshape(13, n_segments, frames_per_segment).mean(axis=2)

    # Save the averaged MFCCs to a CSV file
    output_filename = os.path.join(process_folder, f"{os.path.splitext(audio_files_sorted[i])[0]}_processed_mfcc.csv")
    df = pd.DataFrame(mfccs_avg.T)  # Transpose to (n_segments, 13) before saving
    df.to_csv(output_filename, index=False, header=True)
    
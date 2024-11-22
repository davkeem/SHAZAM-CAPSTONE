import numpy as np
import pandas as pd
import os
import librosa

# Paths to the audio folder and output folder
audio_folder = r"C:\Users\ulalenepi\Downloads\Wav"
process_folder = r"C:\Users\ulalenepi\OneDrive - University of Alaska\Documents\GitHub\SHAZAM-CAPSTONE\processed_chroma"

# Get and sort audio file names
audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]
audio_files_sorted = sorted(audio_files, key=lambda x: int(x.split('_')[0]))
audio_paths = [os.path.join(audio_folder, f) for f in audio_files_sorted]

# Loop over all audio files
for i, file in enumerate(audio_paths):

    # Load the audio file
    y, sr = librosa.load(file, sr=None)

    # Extract Chroma features with default hop length
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)

    # Get hop length and frame duration
    hop_length = 512  # default in librosa
    frame_duration = hop_length / sr  # Duration of each frame in seconds

    # Calculate the number of frames per 0.5 seconds
    frames_per_segment = int(0.5 / frame_duration)
    print(f"Frames per segment: {frames_per_segment}")

    # Trim to ensure full segments, if necessary
    n_segments = chroma.shape[1] // frames_per_segment
    print(f"Chroma shape: {chroma.shape[1]} frames, {n_segments} segments")
    chroma_trimmed = chroma[:, :n_segments * frames_per_segment]

    # Reshape and average over each segment
    chroma_avg = chroma_trimmed.reshape(12, n_segments, frames_per_segment).mean(axis=2)

    # Save the averaged chroma features to a CSV file
    output_filename = os.path.join(process_folder, f"{os.path.splitext(audio_files_sorted[i])[0]}_processed_chroma.csv")
    df = pd.DataFrame(chroma_avg.T)  # Transpose to (n_segments, 12) before saving
    df.to_csv(output_filename, index=False, header=True)

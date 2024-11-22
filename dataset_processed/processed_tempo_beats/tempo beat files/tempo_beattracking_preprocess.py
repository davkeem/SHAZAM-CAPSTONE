import numpy as np
import pandas as pd
import os
import librosa

# Paths for input and output
audio_folder = r"C:\Users\ulalenepi\Downloads\Wav"
audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]
audio_files_sorted = sorted(audio_files, key=lambda x: int(x.split('_')[0]))
audio_paths = [os.path.join(audio_folder, f) for f in audio_files_sorted]

process_folder = r"C:\Users\ulalenepi\OneDrive - University of Alaska\Documents\GitHub\SHAZAM-CAPSTONE\processed_tempo_beats"

# Ensure the output folder exists
os.makedirs(process_folder, exist_ok=True)

# Loop over all audio files
for i, file in enumerate(audio_paths):

    # Load the audio file
    y, sr = librosa.load(file, sr=None)

    # Perform tempo and beat tracking
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    # Convert beat frames to timestamps
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Create a DataFrame to save tempo and beat times
    beat_times_df = pd.DataFrame({"Beat Times (s)": beat_times})  # Each beat time in a separate row
    tempo_df = pd.DataFrame({"Tempo (BPM)": [tempo] * len(beat_times)})  # Repeat tempo for each beat

    # Merge tempo and beat times
    merged_df = pd.concat([tempo_df, beat_times_df], axis=1)

    # Save the tempo and beat times to a CSV file
    output_filename = os.path.join(process_folder, f"{os.path.splitext(audio_files_sorted[i])[0]}_processed_tempo_beats.csv")
    merged_df.to_csv(output_filename, index=False, header=True)

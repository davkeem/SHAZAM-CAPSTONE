import os
import glob
import re
import pandas as pd

# Paths for the processed data
hist_folder = r'C:\Users\ulalenepi\OneDrive - University of Alaska\Documents\GitHub\SHAZAM-CAPSTONE\hist_data'
tempo_beats_folder = r'C:\Users\ulalenepi\OneDrive - University of Alaska\Documents\GitHub\SHAZAM-CAPSTONE\processed_tempo_beats'

# Output folder
ml_data_folder = r'C:\Users\ulalenepi\OneDrive - University of Alaska\Documents\GitHub\SHAZAM-CAPSTONE\processed_tempo_beats\ml_processed_data'
os.makedirs(ml_data_folder, exist_ok=True)

# Collect file paths and sort them
hist_paths = glob.glob(os.path.join(hist_folder, '*.csv'))
hist_paths.sort(key=lambda x: int(re.search(r'song(\d+)', os.path.basename(x)).group(1)))

tempo_beats_paths = glob.glob(os.path.join(tempo_beats_folder, '*.csv'))
tempo_beats_paths.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))

# Merge data
for i in range(len(hist_paths)):
    # Load histogram and tempo/beat tracking data
    hist_data = pd.read_csv(hist_paths[i])
    tempo_beats_data = pd.read_csv(tempo_beats_paths[i])

    # Ensure tempo/beat tracking data has as many rows as hist_data
    t = len(tempo_beats_data)
    h = len(hist_data)

    if t < h:
        # Pad tempo/beat data to match histogram rows
        padding = pd.DataFrame(0, index=range(h - t), columns=tempo_beats_data.columns)
        tempo_beats_data = pd.concat([tempo_beats_data, padding], ignore_index=True)
    elif t > h:
        # Trim tempo/beat data to match histogram rows
        tempo_beats_data = tempo_beats_data.iloc[:h]

    # Merge histogram data with tempo/beat tracking data
    merged_data = pd.concat([
        hist_data.reset_index(drop=True),
        tempo_beats_data.reset_index(drop=True)
    ], axis=1)

    # Save the merged data
    output_filename = f"{os.path.basename(hist_paths[i]).split('_')[0]}_hist_tempo_beats.csv"
    output_path = os.path.join(ml_data_folder, output_filename)
    merged_data.to_csv(output_path, index=False)

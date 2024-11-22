import os
import glob
import re
import numpy as np
import pandas as pd

# Folders for input and output data
onset_folder = r'C:\Users\ulalenepi\OneDrive - University of Alaska\Documents\GitHub\SHAZAM-CAPSTONE\dataset_processed\processed_onset'
hist_folder = r'C:\Users\ulalenepi\OneDrive - University of Alaska\Documents\GitHub\SHAZAM-CAPSTONE\hist_data'

onset_paths = glob.glob(os.path.join(onset_folder, '*.csv'))
onset_paths.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))

hist_paths = glob.glob(os.path.join(hist_folder, '*.csv'))
hist_paths.sort(key=lambda x: int(re.search(r'song(\d+)', os.path.basename(x)).group(1)))

ml_data_folder = r'C:\Users\ulalenepi\OneDrive - University of Alaska\Documents\GitHub\SHAZAM-CAPSTONE\dataset_processed\processed_onset\ml_processed_data'

# Ensure the output folder exists
os.makedirs(ml_data_folder, exist_ok=True)

# Loop through the histogram and onset files
for i in range(len(hist_paths)):
    hist_data = pd.read_csv(hist_paths[i])
    onset_data = pd.read_csv(onset_paths[i])

    h = len(hist_data)
    o = len(onset_data)

    # Trim or pad data to match the number of rows
    if h > o:
        # Pad onset_data with zeros to match the length of hist_data
        padding = pd.DataFrame(0, index=range(h - o), columns=onset_data.columns)
        onset_data = pd.concat([onset_data, padding], ignore_index=True)
    else:
        # Trim onset_data to the length of hist_data
        onset_data = onset_data.iloc[:h]

    # Merge hist_data and onset_data along columns
    merged_data = pd.concat([hist_data.reset_index(drop=True), onset_data.reset_index(drop=True)], axis=1)

    # Save the merged data
    output_filename = f"{os.path.basename(hist_paths[i]).split('_')[0]}_onset_hist.csv"
    output_path = os.path.join(ml_data_folder, output_filename)
    merged_data.to_csv(output_path, index=False)

print("Onset detection and histogram merging completed.")

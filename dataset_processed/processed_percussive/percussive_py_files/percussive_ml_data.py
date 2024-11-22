import os
import glob
import re
import pandas as pd

# Folders for histogram and percussive CSVs
hist_folder = r'C:\Users\ulalenepi\OneDrive - University of Alaska\Documents\GitHub\SHAZAM-CAPSTONE\hist_data'
percussive_folder = r'C:\Users\ulalenepi\OneDrive - University of Alaska\Documents\GitHub\SHAZAM-CAPSTONE\dataset_processed\processed_percussive'

# Get the paths for all CSV files in the folders
percussive_paths = glob.glob(os.path.join(percussive_folder, '*.csv'))
hist_paths = glob.glob(os.path.join(hist_folder, '*.csv'))

# Sort the file paths
percussive_paths.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))
hist_paths.sort(key=lambda x: int(re.search(r'song(\d+)', os.path.basename(x)).group(1)))

# Folder to save the merged data for percussive
percussive_ml_data_folder = r'C:\Users\ulalenepi\OneDrive - University of Alaska\Documents\GitHub\SHAZAM-CAPSTONE\dataset_processed\processed_percussive\ml_processed_data'

# Create output directory if it doesn't exist
os.makedirs(percussive_ml_data_folder, exist_ok=True)

# Loop over each histogram file and process the corresponding data
for i in range(len(hist_paths)):
    hist_data = pd.read_csv(hist_paths[i])
    percussive_data = pd.read_csv(percussive_paths[i])

    h = len(hist_data)
    p = len(percussive_data)

    # Pad or trim Percussive data
    if h > p:
        padding = pd.DataFrame(0, index=range(h - p), columns=percussive_data.columns)
        percussive_data = pd.concat([percussive_data, padding], ignore_index=True)
    else:
        percussive_data = percussive_data.iloc[:h]

    # Merge hist_data with percussive_data
    merged_percussive_data = pd.concat([hist_data.reset_index(drop=True), percussive_data.reset_index(drop=True)], axis=1)

    # Save the merged percussive data to a CSV file
    percussive_output_filename = f"{os.path.basename(hist_paths[i]).split('_')[0]}_hist_percussive.csv"
    percussive_output_path = os.path.join(percussive_ml_data_folder, percussive_output_filename)
    merged_percussive_data.to_csv(percussive_output_path, index=False)

import os
import glob
import re
import pandas as pd

# Folders for histogram and harmonic CSVs
hist_folder = r'C:\Users\ulalenepi\OneDrive - University of Alaska\Documents\GitHub\SHAZAM-CAPSTONE\hist_data'
harmonic_folder = r'C:\Users\ulalenepi\OneDrive - University of Alaska\Documents\GitHub\SHAZAM-CAPSTONE\dataset_processed\processed_harmonic'

# Get the paths for all CSV files in the folders
harmonic_paths = glob.glob(os.path.join(harmonic_folder, '*.csv'))
hist_paths = glob.glob(os.path.join(hist_folder, '*.csv'))

# Sort the file paths
harmonic_paths.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))
hist_paths.sort(key=lambda x: int(re.search(r'song(\d+)', os.path.basename(x)).group(1)))

# Folder to save the merged data for harmonic
harmonic_ml_data_folder = r'C:\Users\ulalenepi\OneDrive - University of Alaska\Documents\GitHub\SHAZAM-CAPSTONE\dataset_processed\processed_harmonic\ml_processed_data'

# Create output directory if it doesn't exist
os.makedirs(harmonic_ml_data_folder, exist_ok=True)

# Loop over each histogram file and process the corresponding data
for i in range(len(hist_paths)):
    hist_data = pd.read_csv(hist_paths[i])
    harmonic_data = pd.read_csv(harmonic_paths[i])

    h = len(hist_data)
    ha = len(harmonic_data)

    # Pad or trim Harmonic data
    if h > ha:
        padding = pd.DataFrame(0, index=range(h - ha), columns=harmonic_data.columns)
        harmonic_data = pd.concat([harmonic_data, padding], ignore_index=True)
    else:
        harmonic_data = harmonic_data.iloc[:h]

    # Merge hist_data with harmonic_data
    merged_harmonic_data = pd.concat([hist_data.reset_index(drop=True), harmonic_data.reset_index(drop=True)], axis=1)

    # Save the merged harmonic data to a CSV file
    harmonic_output_filename = f"{os.path.basename(hist_paths[i]).split('_')[0]}_hist_harmonic.csv"
    harmonic_output_path = os.path.join(harmonic_ml_data_folder, harmonic_output_filename)
    merged_harmonic_data.to_csv(harmonic_output_path, index=False)

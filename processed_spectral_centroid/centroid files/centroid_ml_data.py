import os
import glob
import re
import pandas as pd

# Paths to histogram and spectral centroid data folders
centroid_folder = r'C:\Users\ulalenepi\OneDrive - University of Alaska\Documents\GitHub\SHAZAM-CAPSTONE\processed_spectral_centroid'
hist_folder = r'C:\Users\ulalenepi\OneDrive - University of Alaska\Documents\GitHub\SHAZAM-CAPSTONE\hist_data'

# Output folder for merged data
ml_data_folder = r'C:\Users\ulalenepi\OneDrive - University of Alaska\Documents\GitHub\SHAZAM-CAPSTONE\processed_spectral_centroid\ml_processed_data'
os.makedirs(ml_data_folder, exist_ok=True)

# Get all spectral centroid CSV files
centroid_paths = glob.glob(os.path.join(centroid_folder, '*.csv'))
centroid_paths.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))

# Get all histogram CSV files
hist_paths = glob.glob(os.path.join(hist_folder, '*.csv'))
hist_paths.sort(key=lambda x: int(re.search(r'song(\d+)', os.path.basename(x)).group(1)))

# Merge corresponding files
for i in range(len(hist_paths)):
    # Read histogram and spectral centroid data
    hist_data = pd.read_csv(hist_paths[i])
    centroid_data = pd.read_csv(centroid_paths[i])

    h = len(hist_data)  # Number of rows in histogram data
    c = len(centroid_data)  # Number of rows in spectral centroid data

    # Align row counts by trimming or padding the centroid data
    if h > c:
        # Pad centroid_data with zeros to match the length of hist_data
        padding = pd.DataFrame(0, index=range(h - c), columns=centroid_data.columns)
        centroid_data = pd.concat([centroid_data, padding], ignore_index=True)
    else:
        # Trim centroid_data to match the length of hist_data
        centroid_data = centroid_data.iloc[:h]

    # Merge histogram and spectral centroid data along columns
    merged_data = pd.concat([hist_data.reset_index(drop=True), centroid_data.reset_index(drop=True)], axis=1)

    # Save the merged data
    output_filename = f"{os.path.basename(hist_paths[i]).split('_')[0]}_centroid_hist.csv"
    output_path = os.path.join(ml_data_folder, output_filename)
    merged_data.to_csv(output_path, index=False)

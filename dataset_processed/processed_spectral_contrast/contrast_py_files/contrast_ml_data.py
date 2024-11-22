import os
import glob
import re
import pandas as pd

# Paths to spectral contrast and histogram data
contrast_folder = r'C:\Users\ulalenepi\OneDrive - University of Alaska\Documents\GitHub\SHAZAM-CAPSTONE\processed_spectral_contrast'
hist_folder = r'C:\Users\ulalenepi\OneDrive - University of Alaska\Documents\GitHub\SHAZAM-CAPSTONE\hist_data'

# Output folder for merged data
ml_data_folder = r'C:\Users\ulalenepi\OneDrive - University of Alaska\Documents\GitHub\SHAZAM-CAPSTONE\processed_spectral_contrast\ml_processed_data'
os.makedirs(ml_data_folder, exist_ok=True)

# Get all spectral contrast CSV files
contrast_paths = glob.glob(os.path.join(contrast_folder, '*.csv'))
contrast_paths.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))

# Get all histogram CSV files
hist_paths = glob.glob(os.path.join(hist_folder, '*.csv'))
hist_paths.sort(key=lambda x: int(re.search(r'song(\d+)', os.path.basename(x)).group(1)))

# Merge corresponding files
for i in range(len(hist_paths)):
    # Read histogram and spectral contrast data
    hist_data = pd.read_csv(hist_paths[i])
    contrast_data = pd.read_csv(contrast_paths[i])

    h = len(hist_data)  # Number of rows in histogram data
    c = len(contrast_data)  # Number of rows in spectral contrast data

    # Align row counts by trimming or padding the contrast data
    if h > c:
        # Pad contrast_data with zeros to match the length of hist_data
        padding = pd.DataFrame(0, index=range(h - c), columns=contrast_data.columns)
        contrast_data = pd.concat([contrast_data, padding], ignore_index=True)
    else:
        # Trim contrast_data to match the length of hist_data
        contrast_data = contrast_data.iloc[:h]

    # Merge histogram and spectral contrast data along columns
    merged_data = pd.concat([hist_data.reset_index(drop=True), contrast_data.reset_index(drop=True)], axis=1)

    # Save the merged data
    output_filename = f"{os.path.basename(hist_paths[i]).split('_')[0]}_contrast_hist.csv"
    output_path = os.path.join(ml_data_folder, output_filename)
    merged_data.to_csv(output_path, index=False)

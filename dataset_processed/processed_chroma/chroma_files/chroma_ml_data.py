import os
import glob
import re
import pandas as pd

# Paths to chroma and histogram data
chroma_folder = r'C:\Users\ulalenepi\OneDrive - University of Alaska\Documents\GitHub\SHAZAM-CAPSTONE\processed_chroma'
hist_folder = r'C:\Users\ulalenepi\OneDrive - University of Alaska\Documents\GitHub\SHAZAM-CAPSTONE\hist_data'

# Output folder for merged data
ml_data_folder = r'C:\Users\ulalenepi\OneDrive - University of Alaska\Documents\GitHub\SHAZAM-CAPSTONE\processed_chroma\ml_processed_data'
os.makedirs(ml_data_folder, exist_ok=True)

# Get all chroma CSV files
chroma_paths = glob.glob(os.path.join(chroma_folder, '*.csv'))
chroma_paths.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))

# Get all histogram CSV files
hist_paths = glob.glob(os.path.join(hist_folder, '*.csv'))
hist_paths.sort(key=lambda x: int(re.search(r'song(\d+)', os.path.basename(x)).group(1)))

# Merge corresponding files
for i in range(len(hist_paths)):
    # Read histogram and chroma data
    hist_data = pd.read_csv(hist_paths[i])
    chroma_data = pd.read_csv(chroma_paths[i])

    h = len(hist_data)  # Number of rows in histogram data
    c = len(chroma_data)  # Number of rows in chroma data

    # Align row counts by trimming or padding the chroma data
    if h > c:
        # Pad chroma_data with zeros to match the length of hist_data
        padding = pd.DataFrame(0, index=range(h - c), columns=chroma_data.columns)
        chroma_data = pd.concat([chroma_data, padding], ignore_index=True)
    else:
        # Trim chroma_data to match the length of hist_data
        chroma_data = chroma_data.iloc[:h]

    # Merge histogram and chroma data along columns
    merged_data = pd.concat([hist_data.reset_index(drop=True), chroma_data.reset_index(drop=True)], axis=1)

    # Save the merged data
    output_filename = f"{os.path.basename(hist_paths[i]).split('_')[0]}_chroma_hist.csv"
    output_path = os.path.join(ml_data_folder, output_filename)
    merged_data.to_csv(output_path, index=False)

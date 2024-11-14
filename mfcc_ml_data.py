import os
import glob
import re
import numpy as np
import tensorflow as tf
import pandas as pd

mfcc_folder = r'C:\Users\david\fall2024\capstone\SHAZAM-CAPSTONE\librosa_dataset\mfcc\processed_mfcc'
hist_folder = r'C:\Users\david\fall2024\capstone\SHAZAM-CAPSTONE\histograms\hist_data'

mfcc_paths = glob.glob(os.path.join(mfcc_folder, '*.csv'))
mfcc_paths.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))

hist_paths = glob.glob(os.path.join(hist_folder, '*.csv'))
hist_paths.sort(key=lambda x: int(re.search(r'song(\d+)', os.path.basename(x)).group(1)))

ml_data_folder = r'C:\Users\david\fall2024\capstone\SHAZAM-CAPSTONE\ml_processed_data'

for i in range(len(hist_paths)):
    hist_data = pd.read_csv(hist_paths[i])
    mfcc_data = pd.read_csv(mfcc_paths[i])

    h = len(hist_data)
    m = len(mfcc_data)

    # Trim of pad data to match the number of rows
    if h > m:
        # Pad mfcc_data with zeros to match the length of hist_data
        padding = pd.DataFrame(0, index=range(h - m), columns=mfcc_data.columns)
        mfcc_data = pd.concat([mfcc_data, padding], ignore_index=True)
    else:
        # Trim mfcc_data to the length of hist_data
        mfcc_data = mfcc_data.iloc[:h]

    # Merge hist_data and mfcc_data along columns
    merged_data = pd.concat([hist_data.reset_index(drop=True), mfcc_data.reset_index(drop=True)], axis=1)

    # Save the merged data
    output_filename = f"{os.path.basename(hist_paths[i]).split('_')[0]}_mfcc_hist.csv"
    output_path = os.path.join(ml_data_folder, output_filename)
    merged_data.to_csv(output_path, index=False)


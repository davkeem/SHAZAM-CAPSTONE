import numpy as np
import tensorflow as tf
import pandas as pd
import os
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Concatenate, Flatten
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

mfccs_path = r'C:\Users\david\fall2024\capstone\SHAZAM-CAPSTONE\librosa_dataset\mfcc\MFCC_Data\MFCC_Data'
hist_path = r'C:\Users\david\fall2024\capstone\SHAZAM-CAPSTONE\histograms\histogram_data'

def load_csv_files(directory):
    data = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.csv'):
            file_path = os.path.join(directory, file_name)
            df = pd.read_csv(file_path)
            data.append(df)
    return pd.concat(data, ignore_index=True)

# Load MFCC and histogram data
mfcc_data = load_csv_files(mfccs_path) # len = 386766
hist_data = load_csv_files(hist_path) # len = 4446


#X_train, X_test, y_train, y_test = train_test_split(mfcc_data, hist_data, test_size=0.2, random_state=0)

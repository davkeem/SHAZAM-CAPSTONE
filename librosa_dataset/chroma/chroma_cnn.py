import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Define the directory path
base_dir = r'C:\Users\ulalenepi\OneDrive - University of Alaska\Documents\GitHub\SHAZAM-CAPSTONE\librosa_dataset\chroma\Chroma_Data'

# List of CSV filenames
csv_files = [
    '01_uptown_funk_chroma.csv', '02_thinking_out_loud_chroma.csv', '03_see_you_again_chroma.csv',
    '04_trap_queen_chroma.csv', '05_sugar_chroma.csv', '06_shut_up_and_dance_chroma.csv',
    '07_blank_space_chroma.csv', '08_watch_me_chroma.csv', '09_earned_it_chroma.csv',
    '10_the_hills_chroma.csv', '11_cheerleader_chroma.csv', '12_cant_feel_my_face_chroma.csv',
    '13_love_me_like_you_do_chroma.csv', '14_take_me_to_church_chroma.csv', '16_lean_on_chroma.csv',
    '17_want_to_want_me_chroma.csv', '18_shake_it_off_chroma.csv', '19_where_are_u_now_chroma.csv',
    '20_fight_song_chroma.csv', '21_679_chroma.csv'
]

# Define the path where the model should be saved
save_path = r'C:\Users\ulalenepi\OneDrive - University of Alaska\Documents\GitHub\SHAZAM-CAPSTONE\librosa_dataset\chroma\Chroma_Data\my_model.keras'

# Set target shape for rows and columns (time steps x chroma features)
TARGET_ROWS = 500   # Set based on your data characteristics
TARGET_COLS = 12    # Typically 12 for chroma features

# Load CSV Files with padding/truncating
def load_csv_data(base_dir, csv_files, target_rows=TARGET_ROWS, target_cols=TARGET_COLS):
    print("Loading CSV data with padding/truncating...")
    data = []

    for filename in csv_files:
        file_path = os.path.join(base_dir, filename)
        print(f"Loading file: {filename}")
        df = pd.read_csv(file_path)

        # Check if file has the expected number of columns
        if df.shape[1] != target_cols:
            print(f"Warning: {filename} has {df.shape[1]} columns, expected {target_cols}. Skipping this file.")
            continue

        # Pad or truncate rows to target rows
        array_data = df.values
        if array_data.shape[0] < target_rows:
            # Pad with zeros if there are fewer rows
            padding = np.zeros((target_rows - array_data.shape[0], target_cols))
            array_data = np.vstack([array_data, padding])
            print(f"Padded {filename} to shape {array_data.shape}")
        elif array_data.shape[0] > target_rows:
            # Truncate if there are more rows
            array_data = array_data[:target_rows, :]
            print(f"Truncated {filename} to shape {array_data.shape}")

        data.append(array_data)

    print("CSV data loading complete.")
    return np.array(data)

# Preprocess Data
def preprocess_data(data):
    print("Preprocessing data...")
    num_samples, num_time_steps, num_features = data.shape
    data_flat = data.reshape(-1, num_features)

    scaler = StandardScaler()
    data_flat = scaler.fit_transform(data_flat)

    # Change back to original 3D shape
    processed_data = data_flat.reshape(num_samples, num_time_steps, num_features)
    print("Data preprocessing complete.")
    return processed_data

# Define CNN Model
def create_cnn_model(input_shape):
    print("Creating CNN model...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    print("CNN model created.")
    return model

if __name__ == "__main__":
    print("Starting program...")

    # Load and preprocess data
    data = load_csv_data(base_dir, csv_files)
    if data is None or len(data) == 0:
        print("No data loaded. Exiting program.")
        exit()

    data = preprocess_data(data)

    # Adjust dimensions for CNN input
    print("Adjusting data dimensions for CNN input...")
    data = data[..., np.newaxis]  # Shape: (samples, time_steps, features, 1)

    # Create labels (test example rn: change this based on our actual labels)
    print("Generating random labels for testing...")
    labels = np.random.randint(0, 2, size=(data.shape[0],))

    # Define input shape for the model
    input_shape = (data.shape[1], data.shape[2], 1)

    # Create and compile model
    model = create_cnn_model(input_shape)
    print("Compiling model...")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("Model compilation complete.")

    # Set up early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    print("Starting model training...")
    model.fit(data, labels, epochs=20, batch_size=8, validation_split=0.2, callbacks=[early_stopping])
    print("Model training complete.")

  # Save the model to the specified path
    model.save(save_path)

    print(f"Model saved at: {save_path}")

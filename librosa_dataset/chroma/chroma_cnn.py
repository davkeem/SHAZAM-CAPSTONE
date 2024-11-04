import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Define the directory path
base_dir = r'C:\Users\ulalenepi\OneDrive - University of Alaska\Documents\GitHub\SHAZAM-CAPSTONE\librosa_dataset\chroma\Chroma_Data.zip\Chroma_Data'

# List of CSV filenames
csv_files = [
    '01_uptown_funk.csv', '02_thinking_out_loud.csv', '03_see_you_again.csv',
    '04_trap_queen.csv', '05_sugar.csv', '06_shut_up_and_dance.csv',
    '07_blank_space.csv', '08_watch_me.csv', '09_earned_it.csv',
    '10_the_hills.csv', '11_cheerleader.csv', '12_cant_feel_my_face.csv',
    '13_love_me_like_you_do.csv', '14_take_me_to_church.csv', '15_lean_on.csv',
    '16_want_to_want_me.csv', '17_shake_it_off.csv', '18_where_are_u_now.csv',
    '19_fight_song.csv', '20_679.csv'
]

# Load CSV Files
def load_csv_data(base_dir, csv_files):
    data = []
    for filename in csv_files:
        file_path = os.path.join(base_dir, filename)
        df = pd.read_csv(file_path)
        data.append(df.values)  # each file is time steps x 12 chroma features
    return np.array(data)  

# Preprocess Data
def preprocess_data(data):
    num_samples, num_time_steps, num_features = data.shape
    data_flat = data.reshape(-1, num_features)  

    scaler = StandardScaler()
    data_flat = scaler.fit_transform(data_flat)

    # Change back to original 3D shape
    return data_flat.reshape(num_samples, num_time_steps, num_features)

# Define CNN Model
def create_cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid') 
    ])
    return model


if __name__ == "__main__":
  
    data = load_csv_data(base_dir, csv_files)

    
    data = preprocess_data(data)

    # Adjust dimensions for CNN input
    data = data[..., np.newaxis]  # Shape: (samples, time_steps, features, 1)

    # Create labels (test example rn: change this based on our actual labels)
    labels = np.random.randint(0, 2, size=(data.shape[0],))  

    # Define input shape for the model
    input_shape = (data.shape[1], data.shape[2], 1)

    # Create and compile model
    model = create_cnn_model(input_shape)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    model.fit(data, labels, epochs=20, batch_size=8, validation_split=0.2, callbacks=[early_stopping])

    # Save the model to h5 file
    model.save('chroma_cnn_model.h5')
    print("Model saved as 'chroma_cnn_model.h5'")

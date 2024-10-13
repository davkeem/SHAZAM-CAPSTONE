import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# List of 21 audio file paths
audio_files = [
    r'\Users\jrgroth\Downloads\Wav\01_uptown_funk.wav',
    r'\Users\jrgroth\Downloads\Wav\02_thinking_out_loud.wav',
    r'\Users\jrgroth\Downloads\Wav\03_see_you_again.wav',
    r'\Users\jrgroth\Downloads\Wav\04_trap_queen.wav',
    r'\Users\jrgroth\Downloads\Wav\05_sugar.wav',
    r'\Users\jrgroth\Downloads\Wav\06_shut_up_and_dance.wav',
    r'\Users\jrgroth\Downloads\Wav\07_blank_space.wav',
    r'\Users\jrgroth\Downloads\Wav\08_watch_me.wav',
    r'\Users\jrgroth\Downloads\Wav\09_earned_it.wav',
    r'\Users\jrgroth\Downloads\Wav\10_the_hills.wav',
    r'\Users\jrgroth\Downloads\Wav\11_cheerleader.wav',
    r'\Users\jrgroth\Downloads\Wav\12_cant_feel_my_face.wav',
    r'\Users\jrgroth\Downloads\Wav\13_love_me_like_you_do.wav',
    r'\Users\jrgroth\Downloads\Wav\14_take_me_to_church.wav',
    r'\Users\jrgroth\Downloads\Wav\15_lean_on.wav',
    r'\Users\jrgroth\Downloads\Wav\16_want_to_want_me.wav',
    r'\Users\jrgroth\Downloads\Wav\17_shake_it_off.wav',
    r'\Users\jrgroth\Downloads\Wav\18_where_are_u_now.wav',
    r'\Users\jrgroth\Downloads\Wav\19_fight_song.wav',
    r'\Users\jrgroth\Downloads\Wav\20_fight_song.wav',
    r'\Users\jrgroth\Downloads\Wav\21_679.wav'
]

# Create a directory to save the output PNG files
output_dir = r'\Users\jrgroth\Downloads\Wav\MFCC_Plots'
os.makedirs(output_dir, exist_ok=True)

# Loop through each audio file, compute the MFCCs, and save the graph
for idx, file in enumerate(audio_files):
    # Load the audio file
    y, sr = librosa.load(file, sr=None)

    # Step 2: Extract the 13 MFCCs from the audio file
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Step 3: Create a heatmap to visualize the MFCCs
    plt.figure(figsize=(10, 6))  # Set the figure size for better visualization
    librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap='coolwarm')

    # Add title and labels
    plt.title(f'MFCC Heatmap for File {idx+1}')
    plt.ylabel('MFCC Coefficients')
    plt.xlabel('Time (s)')
    plt.colorbar(format='%+2.0f dB')

    # Save the plot as a PNG file
    output_file = os.path.join(output_dir, f'mfcc_file_{idx+1}.png')
    plt.tight_layout()
    plt.savefig(output_file)

    # Close the plot to free memory for the next one
    plt.close()

print(f"MFCC plots saved to: {output_dir}")

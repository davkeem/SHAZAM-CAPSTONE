import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# List of 20 audio file paths
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
    r'\Users\jrgroth\Downloads\Wav\20_679.wav'
]

# Create a directory to save the output PNG files
output_dir = r'\Users\jrgroth\Downloads\Wav\Spectral_Contrast_Plots'
os.makedirs(output_dir, exist_ok=True)

# Loop through each audio file, compute the spectral contrast, and save the graph
for idx, file in enumerate(audio_files):
    # Load the audio file
    y, sr = librosa.load(file, sr=None)

    # Step 2: Set the hop length and extract spectral contrast
    hop_length = 512  
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)

    # Step 3: Create a figure to visualize the spectral contrast
    plt.figure(figsize=(10, 6))  
    # Display the spectral contrast as a heatmap
    librosa.display.specshow(spectral_contrast, x_axis='time', sr=sr, hop_length=hop_length, cmap='coolwarm')

    # Add color bar and labels
    plt.colorbar(format='%+2.0f')
    plt.title(f'Spectral Contrast Heatmap for File {idx+1}')
    plt.ylabel('Frequency Bands')
    plt.xlabel('Time (s)')
    plt.tight_layout()

    # Save the plot as a PNG file
    output_file = os.path.join(output_dir, f'spectral_contrast_file_{idx+1}.png')
    plt.savefig(output_file)

    # Close the plot to free memory for the next one
    plt.close()

print(f"Spectral contrast plots saved to: {output_dir}")

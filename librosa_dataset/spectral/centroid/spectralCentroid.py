import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# List of 20 paths
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

output_dir = r'\users\jrgroth\Downloads\Wav\Spectral_Centroid_Plots'
os.makedirs(output_dir, exist_ok=True)

# Loop through each audio file, compute the spectral centroid, and save the graph
for idx, file in enumerate(audio_files):
    # Load the audio file
    y, sr = librosa.load(file, sr=None)

    # Step 2: Extract the spectral centroid from the audio file
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    # Step 3: Create a graph to visualize the spectral centroid
    plt.figure(figsize=(10, 6))  # Set the figure size for better visualization
    frames = range(spectral_centroid.shape[1])
    t = librosa.frames_to_time(frames, sr=sr)
    plt.plot(t, spectral_centroid[0], color='b')

    # Add title and labels
    plt.title(f'Spectral Centroid over Time for File {idx+1}')
    plt.ylabel('Spectral Centroid (Hz)')
    plt.xlabel('Time (s)')

    # Save the plot as a PNG file
    output_file = os.path.join(output_dir, f'spectral_centroid_file_{idx+1}.png')
    plt.tight_layout()
    plt.savefig(output_file)

    # Close the plot to free memory for the next one
    plt.close()

print(f"Spectral centroid plots saved to: {output_dir}")

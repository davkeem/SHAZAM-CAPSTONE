import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

# Step 1: Define the paths to your audio files
audio_files = [
    r'C:\Users\ulalenepi\Downloads\Wav\01_uptown_funk.wav',
    r'C:\Users\ulalenepi\Downloads\Wav\02_thinking_out_loud.wav',
    r'C:\Users\ulalenepi\Downloads\Wav\03_see_you_again.wav',
    r'C:\Users\ulalenepi\Downloads\Wav\04_trap_queen.wav',
    r'C:\Users\ulalenepi\Downloads\Wav\05_sugar.wav',
    r'C:\Users\ulalenepi\Downloads\Wav\06_shut_up_and_dance.wav',
    r'C:\Users\ulalenepi\Downloads\Wav\07_blank_space.wav',
    r'C:\Users\ulalenepi\Downloads\Wav\08_watch_me.wav',
    r'C:\Users\ulalenepi\Downloads\Wav\09_earned_it.wav',
    r'C:\Users\ulalenepi\Downloads\Wav\10_the_hills.wav',
    r'C:\Users\ulalenepi\Downloads\Wav\11_cheerleader.wav',
    r'C:\Users\ulalenepi\Downloads\Wav\12_cant_feel_my_face.wav',
    r'C:\Users\ulalenepi\Downloads\Wav\13_love_me_like_you_do.wav',
    r'C:\Users\ulalenepi\Downloads\Wav\14_take_me_to_church.wav',
    r'C:\Users\ulalenepi\Downloads\Wav\15_lean_on.wav',
    r'C:\Users\ulalenepi\Downloads\Wav\16_want_to_want_me.wav',
    r'C:\Users\ulalenepi\Downloads\Wav\17_shake_it_off.wav',
    r'C:\Users\ulalenepi\Downloads\Wav\18_where_are_u_now.wav',
    r'C:\Users\ulalenepi\Downloads\Wav\19_fight_song.wav',
    r'C:\Users\ulalenepi\Downloads\Wav\20_679.wav'
]

# Step 2: Create a directory to save the output PNG files
output_dir = r'C:\Users\ulalenepi\Downloads\Wav\Rhythm_Plots'
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Step 3: Loop through each audio file, extract rhythmic features, and save the graph
for idx, file in enumerate(audio_files):
    # Load the audio file
    y, sr = librosa.load(file, sr=None)

    # Step 4: Extract rhythmic features (onset envelope, tempo, and beats)
    hop_length = 512  # Number of samples per frame

    # Onset envelope: Represents the strength of onsets at each frame
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Calculate tempo and beat frames
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)

    # Step 5: Create a figure for visualization
    plt.figure(figsize=(12, 6))  # Create a 12x6 figure

    # Plot the onset envelope
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, first plot
    plt.plot(librosa.times_like(onset_env, sr=sr, hop_length=hop_length), onset_env, label='Onset Strength')
    plt.xlabel('Time (s)')
    plt.ylabel('Strength')
    plt.title(f'Onset Envelope and Tempo for File {idx + 1}')
    plt.legend(loc='upper right')

    # Plot the detected beats
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, second plot
    times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    plt.vlines(times, 0, 1, color='r', linestyle='--', alpha=0.8, label='Beats')
    plt.xlabel('Time (s)')
    plt.ylabel('Beats')
    plt.legend(loc='upper right')

    # Adjust layout and save the plot
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'rhythm_file_{idx + 1}.png')
    plt.savefig(output_file)  # Save plot as PNG

    # Close the plot to free memory
    plt.close()

print(f"Rhythmic plots saved to: {output_dir}")

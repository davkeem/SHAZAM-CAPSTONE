import librosa
import librosa.display
import matplotlib.pyplot as plt
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

# Step 2: Create a directory to save the output plots
output_dir = r'C:\Users\ulalenepi\Downloads\Wav\Harmonic_Percussive_Plots'
os.makedirs(output_dir, exist_ok=True)

# Step 3: Loop through each audio file, perform harmonic/percussive separation, and save the visualizations
for idx, file in enumerate(audio_files):
    print(f"Processing {file}...")

    # Step 4: Load the audio file
    y, sr = librosa.load(file, sr=None)

    # Step 5: Perform harmonic and percussive source separation using separate functions
    harmonic = librosa.effects.harmonic(y)
    percussive = librosa.effects.percussive(y)

    # Step 6: Create a figure for visualization
    plt.figure(figsize=(12, 8))

    # Plot the original signal
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.5)
    plt.title('Original Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Plot the harmonic component
    plt.subplot(3, 1, 2)
    librosa.display.waveshow(harmonic, sr=sr, color='g', alpha=0.5)
    plt.title('Harmonic Component')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Plot the percussive component
    plt.subplot(3, 1, 3)
    librosa.display.waveshow(percussive, sr=sr, color='r', alpha=0.5)
    plt.title('Percussive Component')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Adjust layout and save the plot
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'hpss_file_{idx + 1}.png')
    plt.savefig(output_file)  # Save plot as PNG
    plt.close()  # Close plot to free memory

print(f"Harmonic/Percussive plots saved to: {output_dir}")

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

# Step 2: Create a directory to save onset plots
output_dir = r'C:\Users\ulalenepi\Downloads\Wav\Onset_Plots'
os.makedirs(output_dir, exist_ok=True)

# Step 3: Loop through each audio file, detect onsets, and save the visualization
for idx, file in enumerate(audio_files):
    print(f"Processing {file}...")

    # Step 4: Load the audio file
    y, sr = librosa.load(file, sr=None)

    # Step 5: Calculate the onset envelope and detect onsets
    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    # Step 6: Create a figure for the onset visualization
    plt.figure(figsize=(12, 6))
    plt.plot(librosa.times_like(onset_env, sr=sr, hop_length=hop_length), onset_env, label='Onset Strength')
    plt.vlines(onset_times, ymin=0, ymax=max(onset_env), color='r', linestyle='--', label='Onsets')
    plt.xlabel('Time (s)')
    plt.ylabel('Onset Strength')
    plt.title(f'Onset Detection for {os.path.basename(file)}')
    plt.legend(loc='upper right')

    # Step 7: Save the plot as a PNG file
    output_file = os.path.join(output_dir, f'onset_file_{idx + 1}.png')
    plt.savefig(output_file)
    plt.close()  # Close plot to free memory

print(f"Onset plots saved to: {output_dir}")

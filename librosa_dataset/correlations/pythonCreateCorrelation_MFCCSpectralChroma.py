import librosa
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

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

# Create directory to save images
output_dir = r'\Users\jrgroth\Downloads\Wav\Correlation_Matrix_Plots'
os.makedirs(output_dir, exist_ok=True)
for idx, file in enumerate(audio_files):

    y, sr = librosa.load(file, sr=None)

    # MFCC 
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # Chroma features 
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    # Spectral contrast 
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    # Spectral centroid 
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    # Combine all features
    combined_features = np.vstack([mfccs, chroma, spectral_contrast, spectral_centroid])
    combined_features = combined_features.T
    # Create DataFrame 
    feature_columns = [f'MFCC_{i+1}' for i in range(13)] + \
                      [f'Chroma_{i+1}' for i in range(12)] + \
                      [f'Contrast_{i+1}' for i in range(spectral_contrast.shape[0])] + \
                      ['Spectral_Centroid']
    feature_df = pd.DataFrame(combined_features, columns=feature_columns)

    correlation_matrix = feature_df.corr()
    
    plt.figure(figsize=(12, 10))
    #Set annot to True to see the correlation values for each square
    sns.heatmap(correlation_matrix, annot=False, cmap='Blues')
    plt.title(f'Feature Correlation Matrix for File {idx+1} ({os.path.basename(file)})')
    output_file = os.path.join(output_dir, f'correlation_matrix_{idx+1}.png')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

print(f"Correlation matrices saved to: {output_dir}")

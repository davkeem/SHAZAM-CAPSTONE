
    features = {}

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features['mfccs_mean'] = np.mean(mfccs, axis=1)
    features['mfccs_std'] = np.std(mfccs, axis=1)

    # Chroma Features
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma_mean'] = np.mean(chroma_stft, axis=1)
    features['chroma_std'] = np.std(chroma_stft, axis=1)

    # Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features['spectral_contrast_mean'] = np.mean(spectral_contrast, axis=1)
    features['spectral_contrast_std'] = np.std(spectral_contrast, axis=1)

    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spectral_centroid_mean'] = np.mean(spectral_centroid)
    features['spectral_centroid_std'] = np.std(spectral_centroid)

    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)

    # Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    features['spectral_rolloff_std'] = np.std(spectral_rolloff)

    # 2. Rhythmic Features
    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = tempo

    # Onset Detection
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    features['onset_env_mean'] = np.mean(onset_env)
    features['onset_env_std'] = np.std(onset_env)

    # Beat Tracking
    _, beats = librosa.beat.beat_track(y=y, sr=sr)
    features['beats'] = len(beats)

    # 3. Harmonic/Percussive Source Separation
    harmonic = librosa.effects.harmonic(y)
    percussive = librosa.effects.percussive(y)
    

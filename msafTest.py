import msaf
import librosa

# Load an audio file
file_path = r'\users\josep\Downloads\Wav\21_679.wav'
# Change this file path to your local file path
y, sr = librosa.load(file_path, sr=None)

# Perform segmentation using MSAF's default configuration
boundaries, labels = msaf.process(file_path)

# Print the detected segment boundaries and labels
print("Segment boundaries (in seconds):", boundaries)
print("Segment labels:", labels)

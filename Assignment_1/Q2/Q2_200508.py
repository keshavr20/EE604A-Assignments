import numpy as np
import librosa

def solution(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    n_fft = 2048
    hop_length = 512
    fmax = 22000
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, fmax=fmax)

    spec_db = librosa.power_to_db(spec, ref=np.max)
    mean_db = np.mean(spec_db)
    threshold = -50.572383880615234
    if mean_db >= threshold:
        return 'metal'
    else:
        return 'cardboard'


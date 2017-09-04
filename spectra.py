import parse_samples as ps
import librosa
import librosa.display

import numpy as np
import matplotlib.pyplot as plt

def get_spectrogram_of_sample(sample):
    print(sample)
    time_series = librosa.core.load(sample.filename)
    print(len(time_series[0]))
    
    mat = librosa.core.stft(time_series[0])
    print(mat)
    print("Frequency size:", mat.shape[0])
    print("Time length:", mat.shape[1])
    
    librosa.display.specshow(librosa.amplitude_to_db(mat, ref=np.max), y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    
    plt.show()

flute_list = ps.get_flute_samples()
get_spectrogram_of_sample(flute_list[0])

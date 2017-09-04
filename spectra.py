import parse_samples as ps
import librosa
import librosa.display

import numpy as np
import matplotlib.pyplot as plt

def get_frequency_matrix(sample):
    """Retrieves a [frequency, time] matrix for a given sample."""
    print(sample)
    time_series = librosa.core.load(sample.filename)
    print(len(time_series[0]))
    return librosa.core.stft(time_series[0])


def get_spectrogram_of_sample(sample):
    """Displays a spectrogram for a given sample. 

    Visual representation of get_frequency_matrix.
    """
    print(sample)
    time_series = librosa.core.load(sample.filename)
    print(len(time_series[0]))
    
    mat = get_frequency_matrix(sample)
    print(mat)
    print("Frequency size:", mat.shape[0])
    print("Time length:", mat.shape[1])
    
    plt.close()
    librosa.display.specshow(librosa.amplitude_to_db(mat, ref=np.max), y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    
    plt.show()


def get_harmonic_spectra(sample):
    """Isolates important frequencies in sample and displays the plot."""
    mat = np.abs(get_frequency_matrix(sample))
    freq_vol = list(np.sum(mat, axis=1))
    x_coord = []
    y_coord = []
    for i in range(len(freq_vol)):
        if freq_vol[i] > 1:
            x_coord.append(64 * i / 6.0)
            y_coord.append(freq_vol[i])
    print(x_coord)
    print(y_coord)
    plt.close()    
    plt.plot(x_coord, y_coord, 'ro')
    plt.show()

flute_list = ps.get_flute_samples()
get_harmonic_spectra(flute_list[0])
# get_spectrogram_of_sample(flute_list[0])

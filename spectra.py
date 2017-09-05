import parse_samples as ps
import librosa
import librosa.display

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

THRESHOLD_FOR_DISTANCE = 100

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

def calculate_spectra_on_frame(mat, val):
    tot_vol = list(np.sum(mat, axis=0))

    vols_norm = list(mat[:,val]) / tot_vol[val]
    x_coord = []
    y_coord = []
    for i in range(len(vols_norm)):
        if vols_norm[i] > 0.002 and i >= 20:
            x_coord.append(64 * i / 6.0)
            y_coord.append(vols_norm[i])
    y_coord /= sum(y_coord)

    x_coord_new = []
    y_coord_new = []
    last = 0
    for i in range(1, len(x_coord)):
        if (x_coord[i] - x_coord[i - 1]) > THRESHOLD_FOR_DISTANCE:
            print(i)
            # print("BLAH")
            tot = 0.0
            weighted_x = 0.0
            for j in range(last, i):
                tot += y_coord[j]
                weighted_x += x_coord[j] * y_coord[j]
            weighted_x /= tot
            x_coord_new.append(weighted_x)
            y_coord_new.append(tot)
            last = i
    tot = 0.0
    weighted_x = 0.0
    for j in range(last, len(x_coord)):
        tot += y_coord[j]
        weighted_x += x_coord[j] * y_coord[j]
    weighted_x /= tot
    x_coord_new.append(weighted_x)
    y_coord_new.append(tot)

    return (x_coord, y_coord, x_coord_new, y_coord_new)

def get_spectra_over_time(sample):
    plt.close()
    mat = np.abs(get_frequency_matrix(sample))
    tot_vol = list(np.sum(mat, axis=0))

    fig = plt.figure()
    fig.subplots_adjust(bottom=0.25)
    ax = fig.add_subplot(111)

    # def calculate_spectra_on_frame(val):
    #     vols_norm = list(mat[:,val]) / tot_vol[val]
    #     # print(len(vols_norm))
    #     x_coord = []
    #     y_coord = []
    #     for i in range(len(vols_norm)):
    #         if vols_norm[i] > 0.002 and i >= 20:
    #             x_coord.append(64 * i / 6.0)
    #             y_coord.append(vols_norm[i])
    #     y_coord /= sum(y_coord)

    #     x_coord_new = []
    #     y_coord_new = []
    #     last = 0
    #     print(x_coord)
    #     for i in range(1, len(x_coord)):
    #         if (x_coord[i] - x_coord[i - 1]) > THRESHOLD_FOR_DISTANCE:
    #             print(i)
    #             # print("BLAH")
    #             tot = 0.0
    #             weighted_x = 0.0
    #             for j in range(last, i):
    #                 tot += y_coord[j]
    #                 weighted_x += x_coord[j] * y_coord[j]
    #             weighted_x /= tot
    #             x_coord_new.append(weighted_x)
    #             y_coord_new.append(tot)
    #             last = i
    #     tot = 0.0
    #     weighted_x = 0.0
    #     for j in range(last, len(x_coord)):
    #         tot += y_coord[j]
    #         weighted_x += x_coord[j] * y_coord[j]
    #     weighted_x /= tot
    #     x_coord_new.append(weighted_x)
    #     y_coord_new.append(tot)

    #     return (x_coord, y_coord, x_coord_new, y_coord_new)

    tup = calculate_spectra_on_frame(mat, 0)
    print(tup)
    ax.plot(tup[0], tup[1], 'ro', tup[2], tup[3], 'bo')
    # ax.plot(tup[2], tup[3], color='blue')

    ax.set_xlim([0, 4000])
    print(64 * mat.shape[1] / 6.0)
    ax.set_ylim([0, 1])

    amp_slider_ax = fig.add_axes([0.20, 0.15, 0.65, 0.03])
    amp_slider = Slider(amp_slider_ax, 'Time', 0, len(tot_vol) - 1, valinit=0, valfmt='%0.0f')
    def sliders_on_changed(val):
        ax.cla()
        ax.set_xlim([0, 4000])
        ax.set_ylim([0, 1])
        tup = calculate_spectra_on_frame(mat, int(val))
        ax.plot(tup[0], tup[1], 'ro', tup[2], tup[3], 'bo')
        # pass
    amp_slider.on_changed(sliders_on_changed)
    plt.show()

flute_list = ps.get_violin_samples()
print(flute_list[50].filename)
# get_harmonic_spectra(flute_list[0])
# get_spectrogram_of_sample(flute_list[0])
get_spectra_over_time(flute_list[50])


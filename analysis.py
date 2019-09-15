import os
from datetime import datetime

import json
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn_porter import Porter
import numpy as np
import matplotlib.pyplot as plt

from MLPClassifierAdjust import export

CHIRP_DURATION = 0.01
SAMPLE_RATE = 44100
F_START = 8000
F_END = 8000
JAVA_SHORT_MAX = 32767
BASE_SOUND_SPEED = 331.3
SOUND_SPEED_COEF = 0.606
CUT_OFF = int(SAMPLE_RATE * (CHIRP_DURATION + 13.0 / BASE_SOUND_SPEED))
SAMPLES_DIR = 'samples'
PREFIX = 'david_porch_2m_'
PREDICTION_CLASSES_FILE_FORMAT = "prediction_classes_{0}"

def get_speed_of_sound(temp):
    return BASE_SOUND_SPEED + SOUND_SPEED_COEF * temp


def get_chirp():
    k = (F_END - F_START) / CHIRP_DURATION
    chirp_size = int(np.ceil(CHIRP_DURATION * SAMPLE_RATE)) + 1
    chirp = np.ndarray(shape=(chirp_size,))
    inc = 1.0 / SAMPLE_RATE
    t = 0.0
    for i in range(len(chirp)):
        if t > CHIRP_DURATION:
            break
        chirp[i] = np.sin(2.0 * np.pi * (F_START * t + 0.5 * k * t ** 2))
        t += inc

    chirp *= np.hanning(len(chirp))
    return chirp * np.full((chirp_size,), JAVA_SHORT_MAX)


def load_samples():
    data = []
    for filename in os.listdir(SAMPLES_DIR):
        with open(os.path.join(SAMPLES_DIR, filename), 'r') as f:
            js = json.load(f)
            line_data = js['cycle'].split('_')
            cc = js['cc']
            line_data.append(calc_dist(cc))
            line_data.append(F_START)
            line_data.append(F_END)
            line_data.append(SAMPLE_RATE)
            line_data.append(CHIRP_DURATION)
            data.append(line_data)

    df = pd.DataFrame(
        data,
        columns=[
            'name',
            'distance',
            'id',
            'calc_dist',
            'f_start',
            'f_end',
            'sample_rate',
            'chirp_duration']
    )
    print(df)


def get_training_data():
    _cross_correlations = []
    dst_labels = []
    for filename in os.listdir(SAMPLES_DIR):
        with open(os.path.join(SAMPLES_DIR, filename), 'r') as f:
            sample = json.load(f)
            dst_label = sample['real_distance']
            cc = sample['cc']
            _cross_correlations.append(cc)
            dst_labels.append(dst_label)

    return np.array(_cross_correlations), np.array(dst_labels)


def load_sample(number):
    with open(os.path.join(SAMPLES_DIR, PREFIX + str(number)), 'r') as f:
        return json.load(f)


def load_cc(number):
    return load_sample(number)['cc']


def load_recording(number):
    return load_sample(number)['recording']


def show_cross_correlation(number):
    data = load_cc(number)
    y = list(float(n) for n in data)
    return get_graph_figure(y, 'CC of {0}'.format(number))


def trim(recording):
    argmax = np.argmax(recording)
    start = max(int(np.floor(argmax - CHIRP_DURATION * SAMPLE_RATE * 0.5)),
                0)
    return recording[start:CUT_OFF + start]


def calc_dist(correlation):
    arr = np.array(correlation)
    peak_width = CHIRP_DURATION * SAMPLE_RATE
    transmitted_peak_index = np.where(arr == np.amax(arr))[0][0]
    return_peak_index = transmitted_peak_index + 220
    for i in range(return_peak_index + 1,
                   min(transmitted_peak_index + 1400, len(arr))):
        j = i - int(peak_width / 4)
        if arr[i] > 0.5e9:
            while arr[i] >= arr[j] and j < min(len(arr) - 1,
                                               i + int(peak_width / 4)):
                j += 1

        if j == len(arr) - 1 or j == i + int(peak_width / 4):
            return_peak_index = i
            break

    if return_peak_index == transmitted_peak_index + 220:
        return 0

    time = (return_peak_index - transmitted_peak_index) * (1.0 / SAMPLE_RATE)
    return get_speed_of_sound(temp=23) * time / 2


def show_recording(number):
    data = load_recording(number)
    y = np.array(list(int(n) for n in data))
    y = trim(y)
    return get_graph_figure(y, 'Recording of {0}'.format(number))


def get_graph_figure(y, title, markers=None):
    x = list(range(len(y)))

    plt.grid(True, axis='y')
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 2, 1])
    ax.set_xlabel('Sample #')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.xaxis.set_ticks(range(0, len(x), 50))
    plt.xticks(rotation=90)
    if not markers:
        ax.plot(x, y)
    else:
        ax.plot(x, y, '-gD', markevery=markers)
    plt.show()
    return fig


def enumerate_distances(distances):
    sorted_distances = sorted(set(distances))
    return zip(sorted_distances, range(len(sorted_distances)))


if __name__ == '__main__':
    print("Loading training data...")
    cross_correlations, distances = get_training_data()  # input cc

    print("Enumerating distances...")
    enumerated_distances = enumerate_distances(distances)

    print("Writing enumerated distances to file...")
    with open(PREDICTION_CLASSES_FILE_FORMAT.format(datetime.utcnow()),
              'w') as f:
        json.dump(enumerated_distances, f)

    MLPClassifier.export = export
    clf = MLPClassifier(solver='lbfgs',
                        hidden_layer_sizes=500,
                        alpha=1e-05,
                        random_state=1)
    print("Learning...")
    clf.fit(cross_correlations, enumerated_distances)

    print("Exporting weights...")
    porter = Porter(clf, language='java')
    java_code = porter.export()

    print("Writing JAVA code to file...")
    with open('MLPClassifier.java', 'w') as f:
        f.write(java_code)

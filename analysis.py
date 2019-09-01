import json
import numpy as np
import matplotlib.pyplot as plt

CHIRP_DURATION = 0.01
SAMPLE_RATE = 44100
F_START = 8000
F_END = 8000
JAVA_SHORT_MAX = 32767
BASE_SOUND_SPEED = 331
CUT_OFF = int(SAMPLE_RATE * (CHIRP_DURATION + 10.0 / BASE_SOUND_SPEED))
RECORDING_FILE_PATH = 'samples/recording_of_{0}'
CC_FILE_PATH = 'samples/cross_correlation_of_{0}'


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


def load_cc(number):
    with open(CC_FILE_PATH.format(number), 'r') as f:
        return json.load(f)['data']


def show_cross_correlation(number):
    data = load_cc(number)
    y = list(float(n) for n in data)
    return get_graph_figure(y, CC_FILE_PATH.format(number))


def trim(recording):
    argmax = np.argmax(recording)
    start = max(int(np.floor(argmax - CHIRP_DURATION * SAMPLE_RATE * 0.5)),
                0)
    return recording[start:CUT_OFF + start]


def load_recording(number):
    with open(RECORDING_FILE_PATH.format(number), 'r') as f:
        return json.load(f)['data']


def show_recording(number):
    data = load_recording(number)
    y = np.array(list(int(n) for n in data))
    y = trim(y)
    return get_graph_figure(y, RECORDING_FILE_PATH.format(number))


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
    ax.plot(x, y, markevery=markers)
    return fig

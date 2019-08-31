import json
import numpy as np
from scipy import signal

import analysis

SAMPLE_NUMBER = 15
RECORDING_FILE_PATH = 'samples/recording_of_{0}'


def load_recording(sample_number):
    with open(RECORDING_FILE_PATH.format(sample_number), 'r') as f:
        return json.load(f)['data']


def trim_recording(recording):
    argmax = np.argmax(recording)
    start_recording = max(
        0,
        int(np.floor(
            argmax - analysis.CHIRP_DURATION * analysis.SAMPLE_RATE * 0.5))
    )
    return recording[start_recording:]


def calc_cross_correlation(data):
    chirp = analysis.get_chirp()
    data = np.array(data)
    data_fft = np.fft.fft(data)
    chirp_fft = np.fft.fft(chirp, n=len(data))
    return np.abs(np.fft.ifft(data_fft * chirp_fft))


def calc_cross_correlation_scipy(data):
    chirp = analysis.get_chirp()
    data = np.array(data)
    return np.abs(signal.hilbert(signal.fftconvolve(chirp, data)))


def main(sample_number):
    data = load_recording(sample_number)
    data = trim_recording(data)
    return calc_cross_correlation(data), calc_cross_correlation_scipy(data)


if __name__ == '__main__':
    main(SAMPLE_NUMBER)

import json
import numpy as np
from scipy import signal

import analysis

SAMPLE_NUMBER = 15
RECORDING_FILE_PATH = 'samples/recording_of_{0}'
CC_FILE_PATH = 'samples/cross_correlation_of_{0}'


def load_recording(sample_number):
    with open(RECORDING_FILE_PATH.format(sample_number), 'r') as f:
        return json.load(f)['data']


def load_cc(sample_number):
    with open(CC_FILE_PATH.format(sample_number), 'r') as f:
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


def get_derivative(data):
    return np.diff(data)


def main(sample_number):
    data = load_recording(sample_number)
    data = trim_recording(data)
    # data_numpy = calc_cross_correlation(data)
    data_scipy = calc_cross_correlation_scipy(data)
    sample_cc = load_cc(sample_number)
    return data_scipy, get_derivative(data_scipy), get_derivative(sample_cc)


if __name__ == '__main__':
    main(SAMPLE_NUMBER)

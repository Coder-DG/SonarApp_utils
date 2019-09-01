import numpy as np
from scipy import signal

import analysis

SAMPLE_NUMBER = 15


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
    data = analysis.load_recording(sample_number)
    data = analysis.trim(data)
    # data_numpy = calc_cross_correlation(data)
    data_scipy = calc_cross_correlation_scipy(data)
    sample_cc = analysis.trim(analysis.load_cc(sample_number))
    return data_scipy, get_derivative(data_scipy), get_derivative(sample_cc)


if __name__ == '__main__':
    main(SAMPLE_NUMBER)

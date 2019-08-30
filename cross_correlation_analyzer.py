import json
import numpy as np

import analysis

SAMPLE_NUMBER = 50
RECORDING_FILE_PATH = 'samples/recording_of_' + str(SAMPLE_NUMBER)


def load_recording():
    with open(RECORDING_FILE_PATH, 'r') as f:
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
    return np.fft.ifft(data_fft * chirp_fft)


def main():
    data = load_recording()
    data = trim_recording(data)
    data = calc_cross_correlation(data)
    return data


if __name__ == '__main__':
    main()

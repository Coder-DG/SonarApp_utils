import json
import numpy as np

from analysis import CHIRP_DURATION, SAMPLE_RATE, get_graph_figure

SAMPLE_NUMBER = 50
RECORDING_FILE_PATH = 'samples/recording_of_' + str(SAMPLE_NUMBER)


def load_recording():
    with open(RECORDING_FILE_PATH, 'r') as f:
        return json.load(f)['data']


def trim_recording(recording):
    argmax = np.argmax(recording)
    start_recording = max(
        0,
        int(np.floor(argmax - CHIRP_DURATION * SAMPLE_RATE * 0.5))
    )
    return recording[start_recording:]


def main():
    recording = load_recording()
    recording = trim_recording(recording)

    return recording


if __name__ == '__main__':
    main()

import numpy as np
from scipy import signal

import analysis

SAMPLE_NUMBER = 15


def get_distances(cc,
                  threshold=2e9,
                  distance=int(
                      analysis.CHIRP_DURATION * analysis.SAMPLE_RATE * 0.5),
                  temp=25):
    speed_of_sound = analysis.get_speed_of_sound(temp)
    peaks, _ = signal.find_peaks(
        cc,
        height=threshold,
        distance=distance)
    peaks = list(peaks)
    if len(peaks) <= 1:
        return None, speed_of_sound, None, peaks
    time_delta = (peaks[1] - peaks[0]) / analysis.SAMPLE_RATE
    distance = time_delta * speed_of_sound / 2.0
    return time_delta, speed_of_sound, distance, peaks


def calc_cross_correlation_scipy(recording, chirp=analysis.get_chirp()):
    recording = np.array(recording)
    return np.abs(signal.hilbert(signal.fftconvolve(chirp, recording)))


def get_derivative(data):
    return np.diff(data)


def get_cc(sample_number):
    return analysis.trim(analysis.load_cc(sample_number))


def main():
    threshold = 1e4

    def get_result(sample_number):
        cc = get_cc(sample_number)
        peaks = list(signal.find_peaks(cc)[0])
        time_delta = (peaks[1] - peaks[0]) / analysis.SAMPLE_RATE
        speed_of_sound = analysis.get_speed_of_sound(25)
        distance = time_delta * speed_of_sound / 2.0
        return cc, time_delta, speed_of_sound, distance / 2.0, peaks

    print_format = "Time {0}\nSpeed of Sound {1}\nDistance {2}"
    for sample_number in range(1, 11):
        cc, time_delta, speed_of_sound, half_distance, peaks = \
            get_result(sample_number)
        print("Peaks: {}".format(["{0:.1e}".format(cc[p]) for p in peaks]))
        print(print_format.format(time_delta, speed_of_sound, half_distance))
        analysis.get_graph_figure(cc, 'Cross Correlation', markers=peaks)
        print("------")


if __name__ == '__main__':
    main()

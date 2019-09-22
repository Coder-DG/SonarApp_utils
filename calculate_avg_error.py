import matplotlib.pyplot as plt
import analysis as an
import cross_correlation_analyzer as cca
import numpy as np
from scipy import signal


def main():
    temp = 23
    sound_of_speed = an.get_speed_of_sound(temp)
    threshold = 10.0 ** 0.85
    holder_distance = 0.35
    frames_to_reach_phone_holder = np.ceil(
        an.SAMPLE_RATE * (holder_distance * 2 / sound_of_speed))
    distance = int(an.CHIRP_DURATION * an.SAMPLE_RATE * 0.5
                   + frames_to_reach_phone_holder)
    prefix = 'david_living_room_shelf'

    start_distance = 1.0
    # vvv Exclusive vvv
    stop_distance = 4.4
    step = 0.5

    def get_avg(dst):
        an.PREFIX = "{0}.{1:.1f}m.".format(prefix, dst)
        scipy_distances = []
        prediction_distances = []
        for sample_number in range(1, 21):
            sample = an.load_sample(sample_number)
            prediction = sample['peaks_prediction']
            if prediction > 0:
                prediction_distances.append(prediction)

            cc = cca.get_cc(sample_number)
            time_delta, speed_of_sound, half_distance, peaks = \
                cca.get_distances(cc, threshold, distance, temp)
            if half_distance:
                scipy_distances.append(half_distance)
        return np.mean(scipy_distances), np.mean(prediction_distances)

    scipy_errors = []
    prediction_errors = []

    distances = np.arange(start_distance, stop_distance, step)
    for dst in distances:
        scipy_avg, prediction_avg = get_avg(dst)

        scipy_error = np.abs(1 - dst / scipy_avg)
        scipy_errors.append(scipy_error)

        prediction_error = np.abs(1 - dst / prediction_avg)
        prediction_errors.append(prediction_error)
        print(
            "Avg distance calc for distance {0:.2f}m:\n"
            "Scipy      - {1:.2f}m\t(error) {3:05.2f}%\n"
            "Prediction - {2:.2f}m\t(error) {4:05.2f}%\n"
            "------------------------------------"
            "".format(dst,
                      scipy_avg,
                      prediction_avg,
                      scipy_error * 100,
                      prediction_error * 100)
        )

    print(
        ">>\tAvg scipy error: {0:05.2f}%\n"
        ">>\tAvg prediction error: {1:05.2f}%"
        "".format(np.mean(scipy_errors) * 100,
                  np.mean(prediction_errors) * 100)
    )


if __name__ == '__main__':
    main()

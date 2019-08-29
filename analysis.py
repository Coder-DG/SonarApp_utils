import matplotlib.pyplot as plt
import json

CHIRP_DURATION = 0.01
SAMPLE_RATE = 44100


def show_cross_correlation(number):
    ccorrelation_file = 'samples/cross_correlation_of_' + str(number)
    show_graph(ccorrelation_file)


def show_recording(number):
    recording_file = 'samples/recording_of_' + str(number)
    show_graph(recording_file)


def show_graph(file_name):
    plt.grid(True, axis='y')
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])

    with open(file_name, 'r') as f:
        data = json.load(f)
        y = list(int(n) for n in data['data'])
    start_y = 0
    for i in y:
        if y[i] == 0:
            start_y += 1
        else:
            break
    y = y[start_y:]
    x = list(range(len(y)))
    ax.plot(x, y)
    ax.set_xlabel('Sample #')
    ax.set_ylabel('Amplitude')
    ax.set_title(file_name)
    fig.show()


def main():
    for i in range(1, 64):
        show_recording(i)
        show_cross_correlation(i)


if __name__ == '__main__':
    main()

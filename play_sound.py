import analysis
import simpleaudio as sa
import numpy as np


def main():
    chirp = analysis.get_chirp().astype(np.int16)

    for i in range(1000):
        play_obj = sa.play_buffer(chirp, 1, 2, analysis.SAMPLE_RATE)
        play_obj.wait_done()


if __name__ == '__main__':
    main()

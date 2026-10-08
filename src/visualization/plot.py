import numpy as np
import matplotlib.pyplot as plt


def plot_time_signal(x, fs):
    t = np.arange(len(x)) / fs

    plt.figure()
    plt.plot(t, x)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Time-Domain Signal")
    plt.grid()
    plt.show()
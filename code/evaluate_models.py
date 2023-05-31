# TODO: add functions: inference on test set, output visualization in comparison to labels
import matplotlib.pyplot as plt
import numpy as np


def plot_evaluation_over_time(data_lists, label_lists, title, evaluation_type):
    steps = np.linspace(1, len(data_lists[0]), len(data_lists[0]))
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle(title, fontsize=16)

    ax1 = fig.add_subplot(1, 1, 1)
    for i, data in enumerate(data_lists):
        ax1.plot(steps, data)
        ax1.scatter(steps, data, label=label_lists[i])
    ax1.set_xlabel("krok uczenia")
    ax1.set_ylabel(evaluation_type)
    ax1.legend()

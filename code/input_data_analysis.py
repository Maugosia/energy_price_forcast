import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import calendar


def plot_temporal_relation(myData, x_ticks_step, yScale):
    plt.figure()
    plt.plot(myData.index, myData["price"], linewidth=0,
             marker='o', markerfacecolor='blue', markersize=0.5)

    plt.tick_params(axis="x", labelrotation=90)
    myLabels = myData["date"][0::x_ticks_step]
    month = 0
    for i in myLabels.index:
        month %= 12
        date = myLabels[i]
        parts = date.split("/")
        myLabels[i] = "/".join([calendar.month_abbr[month+1], parts[2]])
        month += 1

    plt.grid()
    plt.yscale(yScale)
    plt.xticks(np.arange(0, len(myData["date"]), x_ticks_step), labels=myLabels)
    plt.show()


def plot_histogram(myData, binsCount, yScale):
    plt.figure()
    n, bins, patches = plt.hist(myData["price"], bins=binsCount)
    plt.xticks(bins)
    plt.yscale(yScale)
    plt.grid()
    plt.grid(visible=True, which='minor', axis='y', alpha=0.2)
    plt.show()

if __name__ == "__main__":

    path_data = "../day_ahead_data/PGAE_data.csv"
    data = pd.read_csv(path_data, sep=" |,", engine='python', names=["date", "time", "am-pm", "price", "zone"], skiprows=[0])
    
    print("Min value", min(data["price"]))
    print("Max value", max(data["price"]))
    x = np.quantile(data["price"], [0.25, 0.5, 0.75])
    rozstęp = x[2]-x[0];
    wąsD = x[0]-1.5*rozstęp;
    wąsG = x[2]+1.5*rozstęp;
    print("Wąs dolny", wąsD)
    print("Wąs górny", wąsG)
    count = sum(map(lambda x1: x1 > wąsG, data["price"]))
    print(count, "wartości odstających z", len(data["price"]), "(", int(count * 100 / len(data["price"])), "%)")

    #plot_temporal_relation(data, 720, "linear")
    plot_histogram(data, 20, "log")

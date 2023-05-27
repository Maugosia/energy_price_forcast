import pandas as pd
import numpy as np
import torch
from train_gru import train
from pytorch_forecasting import TimeSeriesDataSet
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

def load_data():
    path_data = "day_ahead_data/PGAE_data.csv"
    single_file_data = pd.read_csv(path_data, sep=" |,", engine='python',
                                   names=["date", "time", "am-pm", "price", "zone"], skiprows=[0])

    single_file_data["group"] = 0
    single_file_data["time_idx"] = single_file_data.index

    sc = MinMaxScaler()
    label_sc = MinMaxScaler()
    data = sc.fit_transform(single_file_data["price"].values.reshape(-1,1))
    # Obtaining the Scale for the labels(usage data) so that output can be re-scaled to actual value during evaluation
    label_sc.fit(single_file_data["price"].values.reshape(-1, 1))

    lookback = 90
    inputs = np.zeros((len(single_file_data) - lookback, lookback, single_file_data.shape[1]))
    labels = np.zeros(len(single_file_data) - lookback)

    for i in range(lookback, len(data)):
        inputs[i - lookback] = data[i - lookback:i]
        labels[i - lookback] = data[i, 0]
    inputs = inputs.reshape(-1, lookback, single_file_data.shape[1])
    labels = labels.reshape(-1, 1)

    train_data = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(labels))
    train_loader = DataLoader(train_data, batch_size=64)

    lr = 0.001
    gru_model = train(train_loader, lr)

if __name__ == "__main__":
    load_data()


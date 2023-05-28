import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler


def load_data(data_location_path, batch_size, lookback):
    single_file_data = pd.read_csv(data_location_path, sep=" |,", engine='python',
                                   names=["date", "time", "am-pm", "price", "zone"], skiprows=[0])

    single_file_data["group"] = 0
    single_file_data["time_idx"] = single_file_data.index

    len_all_data = len(single_file_data)
    batch_normalized_len = (np.floor(float(len_all_data)/float(batch_size))) * batch_size
    single_file_data = single_file_data.head(int(batch_normalized_len))

    sc = MinMaxScaler()
    label_sc = MinMaxScaler()
    data = sc.fit_transform(single_file_data["price"].values.reshape(-1, 1))
    # Obtaining the Scale for the labels(usage data) so that output can be re-scaled to actual value during evaluation
    label_sc.fit(single_file_data["price"].values.reshape(-1, 1))

    inputs = np.zeros((len(single_file_data) - lookback, lookback, single_file_data.shape[1]))
    labels = np.zeros(len(single_file_data) - lookback)

    for i in range(lookback, len(data)):
        inputs[i - lookback] = data[i - lookback:i]
        labels[i - lookback] = data[i, 0]
    inputs = inputs.reshape(-1, lookback, single_file_data.shape[1])
    labels = labels.reshape(-1, 1)

    train_data = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(labels))
    train_loader = DataLoader(train_data, batch_size=batch_size)

    return train_loader


if __name__ == "__main__":
    path_data = "day_ahead_data/PGAE_data.csv"
    loader = load_data(path_data, 100)
    print("dataset_size = ", len(loader))

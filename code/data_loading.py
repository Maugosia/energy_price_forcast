import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# TODO:  this function should probably return min_max_scaler object params as well
def load_data(data_location_path, batch_size, lookback, test_proportion=0.2, validation_proportion=0.25):
    # -----------------------------------READ AND PREPARE DATA-------------------------------------------
    single_file_data = pd.read_csv(data_location_path, sep=" |,", engine='python',
                                   names=["date", "time", "am-pm", "price", "zone"], skiprows=[0])

    single_file_data["group"] = 0
    single_file_data["time_idx"] = single_file_data.index

    len_all_data = len(single_file_data)
    batch_normalized_len = (np.floor(float(len_all_data)/float(batch_size))) * batch_size
    single_file_data = single_file_data.head(int(batch_normalized_len))

    # --------------------------------------PREPROCESS DATA-------------------------------------------
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

    # ----------------------------------TRAIN , TEST, VAL SPLIT:-----------------------------------------
    x_train_val, x_test, y_train_val, y_test = train_test_split(inputs, labels,
                                                                test_size=test_proportion, random_state=123)

    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val,
                                                      test_size=validation_proportion, random_state=123)

    len_train_data = y_train.shape[0]
    batch_normalized_len_train = int((np.floor(float(len_train_data)/float(batch_size))) * batch_size)
    y_train = y_train[0:batch_normalized_len_train]
    x_train = x_train[0:batch_normalized_len_train]

    len_val_data = y_val.shape[0]
    batch_normalized_len_val = int((np.floor(float(len_val_data)/float(batch_size))) * batch_size)
    y_val = y_val[0:batch_normalized_len_val]
    x_val = x_val[0:batch_normalized_len_val]

    len_test_data = y_test.shape[0]
    batch_normalized_len_test = int((np.floor(float(len_test_data)/float(batch_size))) * batch_size)
    y_test = y_test[0:batch_normalized_len_test]
    x_test = x_test[0:batch_normalized_len_test]

    print("SHAPE of TRAIN_VAL: ", x_train_val.shape)
    print("SHAPE of TRAIN: ", x_train.shape)
    print("SHAPE of TEST: ", x_test.shape)
    print("SHAPE of VAL: ", x_val.shape)
    print("SHAPE of ALL: ", batch_normalized_len)
    # --------------------------------CREATE DATASETS AND DATALOADERS-----------------------------------
    train_val_test_loaders = []
    train_val_test_data = [[x_train, y_train], [x_val, y_val], [x_test, y_test]]
    for data_pair in train_val_test_data:
        dataset = TensorDataset(torch.from_numpy(data_pair[0]), torch.from_numpy(data_pair[1]))
        data_loader = DataLoader(dataset, batch_size=batch_size)
        train_val_test_loaders.append(data_loader)

    return train_val_test_loaders, label_sc


if __name__ == "__main__":
    path_data = "day_ahead_data/PGAE_data.csv"
    train_loader, val_loader, test_loader, label_transformer = load_data(path_data, 64, 100)
    print("dataset_train size = ", len(train_loader))
    print("dataset_validation size = ", len(val_loader))
    print("dataset_test size = ", len(test_loader))

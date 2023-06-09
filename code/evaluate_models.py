# TODO: add functions: inference on test set, output visualization in comparison to labels
import matplotlib.pyplot as plt
import numpy as np
from globals import DEVICE
import torch
import torch.nn as nn
from data_loading import load_data
from neural_architectures.gru_net import GRUNet
from neural_architectures.lipschitz_net import LipschitzNet
from neural_architectures.lipschitz_net_2 import LipschitzNetWithBatches


def plot_evaluation_over_time(data_lists, label_lists, title, evaluation_type):
    steps = np.linspace(1, len(data_lists[0]), len(data_lists[0]))
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle(title, fontsize=16)

    ax1 = fig.add_subplot(1, 1, 1)
    for i, data in enumerate(data_lists):
        ax1.plot(steps, data)
        ax1.scatter(steps, data, label=label_lists[i])

        # ax1.plot(steps, data, linewidth=0, markersize=0)
        # ax1.scatter(steps, data, label=label_lists[i], s=0.5)
    ax1.set_xlabel("krok uczenia")
    ax1.set_ylabel(evaluation_type)
    ax1.legend()


def inference_on_dataset_GRU(model, test_data_loader, scaler, batch=64):
    criterion = nn.MSELoss()
    avg_loss = 0
    h = model.init_hidden(batch)

    outs = []
    labels = []

    for x, label in test_data_loader:
        h = h.data
        model.zero_grad()

        out, h = model(x.to(DEVICE).float(), h)

        label = torch.from_numpy(scaler.inverse_transform(label))
        out = torch.from_numpy(scaler.inverse_transform(out.detach().numpy()))

        outs.extend(out.detach().tolist())
        labels.extend(label.tolist())

        loss = criterion(out, label.to(DEVICE).float())
        avg_loss += loss.item()

    plot_evaluation_over_time([outs, labels], ["wartości przewidziane przez model", "wartości oczekiwane"],
                              "Wartości zwracane", "wartość")
    print("Average loss: ", avg_loss / len(test_data_loader))
    return avg_loss / len(test_data_loader)


def inference_on_dataset_Lipschitz(model, test_data_loader, scaler):
    criterion = nn.MSELoss()
    avg_loss = 0

    outs = []
    labels = []

    for x, label in test_data_loader:
        model.zero_grad()

        # old_shape = x.shape
        # x = torch.from_numpy(scaler.inverse_transform(x.reshape(-1, 1)).reshape(old_shape))

        out = model(x.to(DEVICE).float())

        label = torch.from_numpy(scaler.inverse_transform(label))
        out = torch.from_numpy(scaler.inverse_transform(out.detach().numpy()))

        outs.extend(out.detach().tolist())
        labels.extend(label.tolist())

        loss = criterion(out, label.to(DEVICE).float())
        avg_loss += loss.item()

    print(len(outs))

    plot_evaluation_over_time([outs, labels], ["wartości przewidziane przez model", "wartości oczekiwane"],
                              "Wartości zwracane", "wartość")
    print("Average loss: ", avg_loss / len(test_data_loader))
    return avg_loss / len(test_data_loader)


def calculateBiasGRU(model, test_data_loader, scaler, batch=64):
    criterion = nn.MSELoss()
    biasSum = 0
    h = model.init_hidden(batch)
    for x, label in test_data_loader:
        h = h.data
        model.zero_grad()

        out, h = model(x.to(DEVICE).float(), h)

        label = torch.from_numpy(scaler.inverse_transform(label))
        out = torch.from_numpy(scaler.inverse_transform(out.detach().numpy()))
        biasSum = biasSum + torch.sum(torch.sub(label, out))
    bias = biasSum / (len(test_data_loader) * batch)
    print("Bias: ", bias)
    return bias


def calculateBiasLipschitz(model, test_data_loader, scaler, batch=128):
    biasSum = 0
    for x, label in test_data_loader:
        out = model(x.to(DEVICE).float())

        label = torch.from_numpy(scaler.inverse_transform(label))
        out = torch.from_numpy(scaler.inverse_transform(out.detach().numpy()))
        biasSum = biasSum + torch.sum(torch.sub(label, out))
    bias = biasSum / (len(test_data_loader) * batch)
    print("Bias: ", bias)
    return bias


def test_GRU():
    # best model intraday
    path_data = "../real_time_data/TH_NP15_data.csv"
    path_model = "trained_models/BEST MODELS/GRU_layers_best_intraday.pt"

    # best model day-ahead
    # path_data = "../day_ahead_data/PGAE_data.csv"
    # path_model = "trained_models/BEST MODELS/GRU_layers_best_day_ahead.pt"

    batch_size = 64
    x_history_length = 128
    [train_loader, val_loader, test_loader], label_transform = load_data(path_data, batch_size, x_history_length)

    input_dim = next(iter(train_loader))[0].shape[2]
    output_dim = 1
    n_layers = 2
    hidden_dim = 256

    GRUmodel = GRUNet(input_dim, hidden_dim, output_dim, n_layers)
    GRUmodel.load_state_dict(torch.load(path_model))
    GRUmodel.eval()
    bias = calculateBiasGRU(GRUmodel, test_loader, label_transform)
    inference_on_dataset_GRU(GRUmodel, test_loader, label_transform)


def test_Lipschitz():
    # best model intraday
    # path_data = "../real_time_data/TH_NP15_data.csv"
    # path_model = "trained_models/BEST MODELS/LIP_layers_best_intraday.pt"

    # best model day-ahead
    path_data = "../day_ahead_data/PGAE_data.csv"
    path_model = "trained_models/BEST MODELS/LIP_layers_best_day_ahead.pt"

    batch_size = 128
    x_history_length = 128
    [train_loader, val_loader, test_loader], label_transform = load_data(path_data, batch_size, x_history_length)

    print("Max {} - Min {}".format(label_transform.data_max_, label_transform.data_min_))

    input_dim = next(iter(train_loader))[0].shape[2]
    output_dim = 1
    hidden_dim = 256

    LipschitzModel = LipschitzNetWithBatches(input_dim, hidden_dim, output_dim, 0.85, 0.85, 0.01, 0.01, dt=0.01)
    LipschitzModel.load_state_dict(torch.load(path_model))
    LipschitzModel.eval()
    LipschitzModel.init_hidden()
    calculateBiasLipschitz(LipschitzModel, test_loader, label_transform)
    inference_on_dataset_Lipschitz(LipschitzModel, test_loader, label_transform)


if __name__ == "__main__":
    # test_GRU()
    test_Lipschitz()

    plt.show()

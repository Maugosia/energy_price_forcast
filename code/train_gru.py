from time import time
import torch
import torch.nn as nn
from gru_net import GRUNet
from data_loading import load_data
from globals import DEVICE


def train(data_loader, learn_rate, hidden_dim=256, epochs=5, batch=64):

    input_dim = next(iter(data_loader))[0].shape[2]
    output_dim = 1
    n_layers = 2

    model = GRUNet(input_dim, hidden_dim, output_dim, n_layers)
    model.to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model.train()
    epoch_times = []
    # Start training loop
    for epoch in range(1, epochs + 1):
        start_time = time()
        h = model.init_hidden(batch)
        avg_loss = 0.
        counter = 0
        for x, label in data_loader:
            counter += 1

            h = h.data

            model.zero_grad()

            out, h = model(x.to(DEVICE).float(), h)
            loss = criterion(out, label.to(DEVICE).float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            # print(epoch, counter)
            if counter % 100 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter,
                                                                                           len(data_loader),
                                                                                           avg_loss / counter))
        current_time = time()
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, epochs, avg_loss / len(data_loader)))
        print("Total Time Elapsed: {} seconds".format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)
    print("Total Training Time: {} seconds".format(str(sum(epoch_times))))
    return model


if __name__ == "__main__":
    # PARAMS
    lr = 0.001
    batch_size = 64
    x_history_length = 64
    path_data = "../day_ahead_data/PGAE_data.csv"

    # LOAD AND TRAIN
    train_loader = load_data(path_data, batch_size, x_history_length)
    gru_model = train(train_loader, lr, batch=64)


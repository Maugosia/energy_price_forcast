import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet




def load_data():
    print(torch.__version__)
    path_data = "day_ahead_data/PGAE_data.csv"
    single_file_data = pd.read_csv(path_data, sep=" |,", engine='python',
                                   names=["date", "time", "am-pm", "price", "zone"], skiprows=[0])

    single_file_data["group"] = 0
    single_file_data["time_idx"] = single_file_data.index
    print(single_file_data.head(10))


    dataset = TimeSeriesDataSet(
        single_file_data,
        time_idx="time_idx",
        target="price",
        group_ids=["group"],
    )
    print(dataset[0])


    dataloader = dataset.to_dataloader()
    x,y = next(iter(dataloader))

    print("x =", x)
    print("\ny =", y)
    print("\nsizes of x =")
    for key, value in x.items():
        print(f"\t{key} = {value.size()}")

if __name__ == "__main__":
    load_data()


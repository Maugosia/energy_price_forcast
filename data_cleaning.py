import pandas as pd
import os


def join_and_clean_day_ahead_data():
    path_data = "data"
    filenames = next(os.walk(path_data), (None, None, []))[2]
    print(filenames)
    print(len(filenames))
    all_data = []
    for i, f in enumerate(filenames):
        single_file_data = pd.read_csv(path_data + "/" + f, sep=" |,", engine='python',
                                       names=["date", "time", "am-pm", "price", "zone"], skiprows=[0])
        all_data.append(single_file_data)

    print(all_data[0].head(10))
    print("\n\n")
    all_dataframe = pd.concat(all_data)
    print("\n\n")
    print(all_dataframe.describe())
    print("\n\n")
    print(all_dataframe.head(100).tail(10))

    zone_names = ["TH_NP15", "TH_SP15", "TH_ZP26"]
    single_zone_dataframes = []
    for zone_name in zone_names:
        single_zone_dataframe = all_dataframe[all_dataframe["zone"] == zone_name]

        print(len(single_zone_dataframe))
        single_zone_dataframe.drop_duplicates(inplace=True, ignore_index=True)
        print(len(single_zone_dataframe), "\n")

        single_zone_dataframes.append(single_zone_dataframe)
        single_zone_dataframe.to_csv(path_data + "/real_time/" + zone_name + "_data.csv")

    all_dataframe.to_csv(path_data + "/real_time/" + "all_data.csv")


if __name__ == "__main__":
    join_and_clean_day_ahead_data()

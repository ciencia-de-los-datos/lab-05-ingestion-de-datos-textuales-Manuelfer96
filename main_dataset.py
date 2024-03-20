import glob
import pandas as pd

COLUMNS = ["phrase", "sentiment"]


def read_paths():
    paths_train = glob.glob(
        "train/*/*.txt", recursive=True, include_hidden=True)
    paths_tests = glob.glob(
        "test/*/*.txt", recursive=True, include_hidden=True)
    return paths_train, paths_tests


def build_dataframe(pahts):
    df = pd.DataFrame([{"phrase": pd.read_csv(path, header=None, index_col=None)[
        0][0], "sentiment": path.split("\\")[1]} for path in pahts])
    return df


def run():

    trains, test = read_paths()
    df_trains = build_dataframe(trains)
    df_test = build_dataframe(test)
    df_trains.to_csv("train_dataset.csv", index=False)
    df_test.to_csv("test_dataset.csv", index=False)


if __name__ == "__main__":
    run()

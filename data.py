import pandas as pd
import numpy as np

SELECTED_NODES = [3120, 3122, 3126, 3180, 4030, 4032, 4034, 4035, 4040, 4043]


def load_raw_excel(file):
    return pd.read_excel(file, sheet_name="Data", engine="xlrd")


def clean_data(df):
    df = df.copy()
    df["SCATS Number"] = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    return df


def extract_flow(df):
    df = df.copy()

    df["SCATS Number"] = pd.to_numeric(df["SCATS Number"], errors="coerce")
    df = df[df["SCATS Number"].isin(SELECTED_NODES)]

    flow_data = df.iloc[:, 36:40].apply(pd.to_numeric, errors="coerce")
    df["flow"] = flow_data.sum(axis=1)

    return df


def load_data(file="Scats Data October 2006.xls"):
    df = load_raw_excel(file)
    df = clean_data(df)
    df = extract_flow(df)
    df = df.dropna(subset=["flow"])
    return df


def create_train_test(df, ratio=0.8):
    df = df.sort_values("SCATS Number")

    split = int(len(df) * ratio)
    train = df.iloc[:split]
    test = df.iloc[split:]

    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)

    print("Train/Test saved")
    return train, test


if __name__ == "__main__":
    df = load_data("Scats Data October 2006.xls")
    create_train_test(df)
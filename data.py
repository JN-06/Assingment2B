"""
Data processing for Traffic Regression Tree + Graph Weight System
"""

import pandas as pd

# Selected SCATS + ONE location each
SELECTED_LOCATIONS = {
    3120: "BURKE_RD N of CANTERBURY_RD",
    3122: "CANTERBURY_RD E of STANHOPE_GV",
    3126: "CANTERBURY_RD E of WARRIGAL_RD",
    3180: "BALWYN_RD S of DONCASTER_RD",
    4030: "BURKE_RD S of DONCASTER_RD",
    4032: "BURKE_RD N of HARP_RD",
    4034: "BURKE_RD N OF WHITEHORSE_RD",
    4035: "BURKE_RD N of MONT ALBERT_RD",
    4040: "BURKE_RD N of RIVERSDALE_RD",
    4043: "BURKE_RD N of TOORAK_RD"
}

# Load Excel file
def load_raw_excel(file):

    df = pd.read_excel(
        file,
        sheet_name="Data",
        engine="xlrd",
        header=1
    )

    # remove hidden spaces
    df.columns = df.columns.str.strip()

    return df

# Clean dataset
def clean_data(df):
    df = df.copy()

    df["SCATS Number"] = pd.to_numeric(df["SCATS Number"], errors="coerce")
    df = df.dropna(subset=["SCATS Number"])

    return df

# Feature engineering + flow extraction
def extract_flow(df):
    df = df.copy()

    # STEP 1: Filter SCATS + exact location
    df = df[
        df.apply(
            lambda row: (
                row["SCATS Number"] in SELECTED_LOCATIONS and
                row["Location"] == SELECTED_LOCATIONS[row["SCATS Number"]]
            ),
            axis=1
        )
    ]

    # STEP 2: Convert date features
    df["Date"] = pd.to_datetime(df["Date"])

    df["day"] = df["Date"].dt.day
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year
    df["day_of_week"] = df["Date"].dt.dayofweek  # Monday=0

    # STEP 3: Target variable (traffic flow)
    df["flow_9to10"] = df[["V36", "V37", "V38", "V39"]].sum(axis=1)

    # STEP 4: Final dataset for ML
    df = df[[
        "SCATS Number",
        "Location",
        "day",
        "month",
        "year",
        "day_of_week",
        "flow_9to10"
    ]]

    return df

# Full pipeline loader
def load_data(file="Scats Data October 2006.xls"):
    df = load_raw_excel(file)
    df = clean_data(df)
    df = extract_flow(df)
    df = df.dropna(subset=["flow_9to10"])
    return df

# Train / Test split
def create_train_test(df, ratio=0.8):
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    split = int(len(df) * ratio)

    train = df.iloc[:split]
    test = df.iloc[split:]

    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)

    print("Train/Test datasets created successfully!")

    return train, test

# Run pipeline
if __name__ == "__main__":
    df = load_data("Scats Data October 2006.xls")
    create_train_test(df)
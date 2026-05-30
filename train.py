import json

import pandas as pd
import joblib
import random
import tensorflow as tf

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense

import numpy as np

# Fix random seed
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Global config 
STEP = 12

NODES_LIST = [
    3120, 3122, 3126, 3180, 4030,
    4032, 4034, 4035, 4040, 4043
]

# Feature engineering
def create_dt_features(df):

    df = df.copy()

    # lag features
    df["flow_t1"] = df["flow_9to10"].shift(1)
    df["flow_t2"] = df["flow_9to10"].shift(2)

    # rolling features
    df["flow_mean_3"] = df["flow_9to10"].rolling(3).mean()
    df["flow_std_3"] = df["flow_9to10"].rolling(3).std()

    # fill missing values created by lag and rolling features
    df = df.bfill()

    return df

# Load dataset
def load_dataset():

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    def extract_features(df):

        df = create_dt_features(df)

        # time features
        if "hour" not in df.columns:
            df["hour"] = 9

        if "day_of_week" not in df.columns:
            df["day_of_week"] = 1

        if "day" not in df.columns:
            df["day"] = 1

        if "month" not in df.columns:
            df["month"] = 10

        # location one hot
        for n in NODES_LIST:
            df[f"Location_{n}"] = (
                df["SCATS Number"] == n
            ).astype(int)

        feature_cols = [

            # time features
            "hour",
            "day_of_week",
            "day",
            "month",

            # traffic features
            "flow_t1",
            "flow_t2",
            "flow_mean_3",
            "flow_std_3"

        ] + [f"Location_{n}" for n in NODES_LIST]

        X = df[feature_cols].values

        y = df["flow_9to10"].values

        return X, y

    X_train, y_train = extract_features(train)
    X_test, y_test = extract_features(test)

    # scale target
    scaler_y = MinMaxScaler()

    y_train = scaler_y.fit_transform(
        y_train.reshape(-1, 1)
    ).flatten()

    y_test = scaler_y.transform(
        y_test.reshape(-1, 1)
    ).flatten()

    # scale features
    scaler_X = MinMaxScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    # save scaler for inference
    joblib.dump(scaler_X, "model/scaler_X.pkl")

    return X_train, X_test, y_train, y_test, scaler_y

# sequence builder
def create_sequence(X, y, step=12):

    X_seq = []
    y_seq = []

    for i in range(len(X) - step):

        # previous timesteps
        sequence = X[i:i + step]

        # target
        target = y[i + step]

        X_seq.append(sequence)
        y_seq.append(target)

    return np.array(X_seq), np.array(y_seq)

# LSTM Model
def build_lstm(input_shape):

    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model

# GRU Model
def build_gru(input_shape):

    model = Sequential([
        GRU(64, input_shape=input_shape),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model

# DECISION TREE
def train_dt(X_train, y_train):

    dt = DecisionTreeRegressor(
        max_depth=6,
        min_samples_leaf=5,
        random_state=42
    )

    dt.fit(X_train, y_train)

    joblib.dump(
        dt,
        "model/decision_tree.pkl"
    )

    # loss log
    preds = dt.predict(X_train)

    errors = np.abs(
        y_train.flatten() - preds
    )

    pd.DataFrame({
        "step": np.arange(len(errors)),
        "absolute_error": errors
    }).to_csv(
        "model/dt_loss.csv",
        index=False
    )

    return dt

# Evaluation
def evaluate_model(y_true, y_pred):

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    nrmse = rmse / (
        y_true.max() - y_true.min()
    )

    return rmse, mae, nrmse

# Display results
def display_predictions(
    model_name,
    actual_scaled,
    predicted_scaled,
    y_scaler
):

    actual = y_scaler.inverse_transform(
        actual_scaled.reshape(-1, 1)
    ).flatten()

    predicted = y_scaler.inverse_transform(
        predicted_scaled.reshape(-1, 1)
    ).flatten()

    errors = np.abs(actual - predicted)

    accuracy = 100 - np.mean(
        errors / actual
    ) * 100

    print("\n===================================")
    print(f"{model_name} PREDICTION RESULTS")
    print("===================================")

    for i in range(10):

        print(
            f"Sample {i+1} | "
            f"Actual: {int(actual[i])} cars | "
            f"Predicted: {int(predicted[i])} cars | "
            f"Error: {int(errors[i])}"
        )

    print(f"\n{model_name} Accuracy: {accuracy:.2f}%")

    return accuracy

# Main
def main():

    X_train, X_test, y_train, y_test, scaler = load_dataset()
    
    # create RNN Sequences
    
    X_train_seq, y_train_seq = create_sequence(
        X_train,
        y_train,
        STEP
    )

    X_test_seq, y_test_seq = create_sequence(
        X_test,
        y_test,
        STEP
    )

    input_shape = (
        X_train_seq.shape[1],
        X_train_seq.shape[2]
    )

    results = []

    # LSTM
    print("\nTraining LSTM...")

    lstm = build_lstm(input_shape)

    hist_lstm = lstm.fit(
        X_train_seq,
        y_train_seq,
        epochs=30,
        batch_size=32,
        verbose=1
    )

    lstm.save("model/lstm.h5")

    pd.DataFrame(
        hist_lstm.history
    ).to_csv(
        "model/lstm_loss.csv",
        index=False
    )

    lstm_pred = lstm.predict(
        X_test_seq,
        verbose=0
    )

    lstm_rmse, lstm_mae, lstm_nrmse = evaluate_model(
        y_test_seq,
        lstm_pred
    )

    lstm_acc = display_predictions(
        "LSTM",
        y_test_seq,
        lstm_pred,
        scaler
    )

    results.append([
        "LSTM",
        lstm_rmse,
        lstm_mae,
        lstm_nrmse,
        lstm_acc
    ])

    # GRU
    print("\nTraining GRU...")

    gru = build_gru(input_shape)

    hist_gru = gru.fit(
        X_train_seq,
        y_train_seq,
        epochs=30,
        batch_size=32,
        verbose=1
    )

    gru.save("model/gru.h5")

    pd.DataFrame(
        hist_gru.history
    ).to_csv(
        "model/gru_loss.csv",
        index=False
    )

    gru_pred = gru.predict(
        X_test_seq,
        verbose=0
    )

    gru_rmse, gru_mae, gru_nrmse = evaluate_model(
        y_test_seq,
        gru_pred
    )

    gru_acc = display_predictions(
        "GRU",
        y_test_seq,
        gru_pred,
        scaler
    )

    results.append([
        "GRU",
        gru_rmse,
        gru_mae,
        gru_nrmse,
        gru_acc
    ])
    
    # DECISION TREE
    print("\nTraining DT...")

    dt = train_dt(
        X_train,
        y_train
    )

    dt_pred = dt.predict(X_test)

    dt_rmse, dt_mae, dt_nrmse = evaluate_model(
        y_test,
        dt_pred
    )

    dt_acc = display_predictions(
        "DT",
        y_test,
        dt_pred,
        scaler
    )

    results.append([
        "DT",
        dt_rmse,
        dt_mae,
        dt_nrmse,
        dt_acc
    ])

    # save results    
    results_df = pd.DataFrame(
        results,
        columns=[
            "model",
            "rmse",
            "mae",
            "nrmse",
            "accuracy"
        ]
    )

    import json

    results_dict = {
        "LSTM": float(lstm_acc),
        "GRU": float(gru_acc),
        "DT": float(dt_acc)
    }

    with open("model/results.json", "w") as f:
        json.dump(results_dict, f, indent=4)

    print("\n==============================")
    print("FINAL RESULTS")
    print("==============================")

    print(results_df)

    joblib.dump(scaler, "model/y_scaler.pkl")

# Run
if __name__ == "__main__":
    main()
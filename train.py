import numpy as np
import pandas as pd
import joblib
import random
import tensorflow as tf

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense

# =========================
# FIX RANDOM SEED
# =========================
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# =========================
# LOAD DATA
# =========================
def load_dataset():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    y_train = train["flow_9to10"].values
    y_test = test["flow_9to10"].values

    scaler = MinMaxScaler()

    y_train = scaler.fit_transform(y_train.reshape(-1, 1))
    y_test = scaler.transform(y_test.reshape(-1, 1))

    return y_train, y_test, scaler


# =========================
# SEQUENCE BUILDER
# =========================
def create_sequence(data, step=12):
    X, y = [], []

    for i in range(len(data) - step):
        X.append(data[i:i + step])
        y.append(data[i + step])

    return np.array(X), np.array(y)


# =========================
# MODELS
# =========================
def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def build_gru(input_shape):
    model = Sequential([
        GRU(64, input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def train_dt(X_train, y_train):
    dt = DecisionTreeRegressor(
        max_depth=8,
        min_samples_leaf=2,
        random_state=42
    )

    dt.fit(
        X_train.reshape(X_train.shape[0], -1),
        y_train
    )

    joblib.dump(dt, "model/decision_tree.pkl")
    return dt

# =========================
# METRICS
# =========================
def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    nrmse = rmse / (y_true.max() - y_true.min())

    print(f"\n{name}")
    print("-" * 30)
    print("RMSE :", round(rmse, 5))
    print("MAE  :", round(mae, 5))
    print("NRMSE:", round(nrmse, 5))

    return rmse, mae, nrmse


# =========================
# MAIN TRAINING PIPELINE
# =========================
def main():

    y_train, y_test, scaler = load_dataset()

    STEP = 12

    # sequence data
    X_train, y_train_seq = create_sequence(y_train, STEP)
    X_test, y_test_seq = create_sequence(y_test, STEP)

    # reshape for LSTM/GRU
    X_train_rnn = X_train.reshape(X_train.shape[0], STEP, 1)
    X_test_rnn = X_test.reshape(X_test.shape[0], STEP, 1)

    # =========================
    # LSTM
    # =========================
    print("\nTraining LSTM...")
    lstm = build_lstm((STEP, 1))
    lstm.fit(X_train_rnn, y_train_seq, epochs=10, batch_size=32, verbose=1)
    lstm.save("model/lstm.h5")

    # =========================
    # GRU
    # =========================
    print("\nTraining GRU...")
    gru = build_gru((STEP, 1))
    gru.fit(X_train_rnn, y_train_seq, epochs=10, batch_size=32, verbose=1)
    gru.save("model/gru.h5")

    # =========================
    # DECISION TREE
    # =========================
    print("\nTraining DT...")
    dt = train_dt(X_train, y_train_seq)

    # =========================
    # PREDICTIONS
    # =========================
    lstm_pred = lstm.predict(X_test_rnn, verbose=0)
    gru_pred = gru.predict(X_test_rnn, verbose=0)

    dt_pred = dt.predict(X_test.reshape(X_test.shape[0], -1))

    # =========================
    # EVALUATION
    # =========================
    print("\n==============================")
    print("MODEL COMPARISON RESULTS")
    print("==============================")

    lstm_rmse, lstm_mae, lstm_nrmse = evaluate_model("LSTM", y_test_seq, lstm_pred)
    gru_rmse, gru_mae,  gru_nrmse = evaluate_model("GRU", y_test_seq, gru_pred)
    dt_rmse, dt_mae, dt_nrmse = evaluate_model("DT", y_test_seq, dt_pred)

    # =========================
    # SUMMARY TABLE (IMPORTANT FOR REPORT)
    # =========================
    print("\n==============================")
    print("FINAL COMPARISON TABLE")
    print("==============================")

    print(f"{'Model':<15}{'RMSE':<15}{'MAE':<15}{'NRMSE':<15}")
    print("-" * 60)

    print(f"LSTM{'':<11}{lstm_rmse:<15.5f}{lstm_mae:<15.5f}{lstm_nrmse:<15.5f}")
    print(f"GRU{'':<12}{gru_rmse:<15.5f}{gru_mae:<15.5f}{gru_nrmse:<15.5f}")
    print(f"DT{'':<13}{dt_rmse:<15.5f}{dt_mae:<15.5f}{dt_nrmse:<15.5f}")

    print("\nALL MODELS SAVED SUCCESSFULLY")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    main()
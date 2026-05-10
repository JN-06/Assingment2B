import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense


# =========================
# LOAD DATA
# =========================
def load_dataset():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    y_train = train["flow"].values
    y_test = test["flow"].values

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


def train_rf(X_train, y_train):
    rf = RandomForestRegressor(
        n_estimators=150,
        random_state=42
    )

    rf.fit(
        X_train.reshape(X_train.shape[0], -1),
        y_train
    )

    joblib.dump(rf, "model/rf.pkl")
    return rf


# =========================
# METRICS
# =========================
def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    print(f"\n{name}")
    print("-" * 30)
    print("RMSE:", round(rmse, 5))
    print("MAE :", round(mae, 5))

    return rmse, mae


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
    # RANDOM FOREST
    # =========================
    print("\nTraining Random Forest...")
    rf = train_rf(X_train, y_train_seq)

    # =========================
    # PREDICTIONS
    # =========================
    lstm_pred = lstm.predict(X_test_rnn, verbose=0)
    gru_pred = gru.predict(X_test_rnn, verbose=0)

    rf_pred = rf.predict(X_test.reshape(X_test.shape[0], -1))

    # =========================
    # EVALUATION
    # =========================
    print("\n==============================")
    print("MODEL COMPARISON RESULTS")
    print("==============================")

    lstm_rmse, lstm_mae = evaluate_model("LSTM", y_test_seq, lstm_pred)
    gru_rmse, gru_mae = evaluate_model("GRU", y_test_seq, gru_pred)
    rf_rmse, rf_mae = evaluate_model("Random Forest", y_test_seq, rf_pred)

    # =========================
    # SUMMARY TABLE (IMPORTANT FOR REPORT)
    # =========================
    print("\n==============================")
    print("FINAL COMPARISON TABLE")
    print("==============================")

    print(f"{'Model':<15}{'RMSE':<15}{'MAE'}")
    print("-" * 40)
    print(f"LSTM{'':<11}{lstm_rmse:<15.5f}{lstm_mae:.5f}")
    print(f"GRU{'':<12}{gru_rmse:<15.5f}{gru_mae:.5f}")
    print(f"RF{'':<13}{rf_rmse:<15.5f}{rf_mae:.5f}")

    print("\nALL MODELS SAVED SUCCESSFULLY")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    main()
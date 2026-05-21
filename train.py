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

    def extract_features(df):         
        df = df.copy()
        df["hour"] = 9  # flow_9to10 always = 9am slot

        df = pd.get_dummies(df, columns=["Location"])  # capital L to match data.py

        feature_cols = ["hour", "day_of_week", "day", "month"] + \
                       [c for c in df.columns if c.startswith("Location_")]
        return df[feature_cols].values, df["flow_9to10"].values

    X_train_meta, y_train = extract_features(train)
    X_test_meta,  y_test  = extract_features(test)

    scaler_y = MinMaxScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test  = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    scaler_X = MinMaxScaler()
    X_train_meta = scaler_X.fit_transform(X_train_meta)
    X_test_meta  = scaler_X.transform(X_test_meta)

    return X_train_meta, X_test_meta, y_train, y_test, scaler_y


# =========================
# SEQUENCE BUILDER
# =========================
def create_sequence(X_meta, y, step=12):
    X, targets = [], []
    for i in range(len(y) - step):
        past_flow = y[i:i + step]               # shape (step,)
        meta      = X_meta[i + step]            # location + date of the TARGET step
        combined  = np.concatenate([past_flow, meta])
        X.append(combined)
        targets.append(y[i + step])
    return np.array(X), np.array(targets)


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


# =========================
# DECISION TREE + LOSS CSV
# =========================
def train_dt(X_train, y_train):
    dt = DecisionTreeRegressor(max_depth=8, min_samples_leaf=2, random_state=42)
    dt.fit(X_train, y_train)   # no reshape needed anymore
    joblib.dump(dt, "model/decision_tree.pkl")
    preds = dt.predict(X_train)
    errors = np.abs(y_train.flatten() - preds)
    pd.DataFrame({"step": np.arange(len(errors)), "absolute_error": errors}).to_csv("model/dt_loss.csv", index=False)
    return dt


# =========================
# METRICS
# =========================
def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    nrmse = rmse / (y_true.max() - y_true.min())
    return rmse, mae, nrmse


# =========================
# DISPLAY PREDICTIONS
# =========================
def display_predictions(model_name, actual_scaled, predicted_scaled, y_scaler):
    actual = y_scaler.inverse_transform(
        actual_scaled.reshape(-1, 1)
    ).flatten()

    predicted = y_scaler.inverse_transform(
        predicted_scaled.reshape(-1, 1)
    ).flatten()

    errors = np.abs(actual - predicted)
    accuracy = 100 - np.mean(errors / actual) * 100

    print(f"\n===================================")
    print(f"{model_name} PREDICTION RESULTS")
    print(f"===================================")

    for i in range(10):
        print(
            f"Sample {i+1} | "
            f"Actual: {int(actual[i])} cars | "
            f"Predicted: {int(predicted[i])} cars | "
            f"Error: {int(errors[i])}"
        )

    print(f"\n{model_name} Accuracy: {accuracy:.2f}%")
    return accuracy


# =========================
# MAIN
# =========================
def main():
    X_train_meta, X_test_meta, y_train, y_test, scaler = load_dataset()
    STEP = 12

    X_train, y_train_seq = create_sequence(X_train_meta, y_train, STEP)
    X_test,  y_test_seq  = create_sequence(X_test_meta,  y_test,  STEP)

    input_dim = X_train.shape[1]  # STEP + number of meta features

    # RNN expects (samples, timesteps, features) — treat full vector as 1 timestep
    X_train_rnn = X_train.reshape(X_train.shape[0], 1, input_dim)
    X_test_rnn  = X_test.reshape(X_test.shape[0],  1, input_dim)

    results = []

    # LSTM
    print("\nTraining LSTM...")
    lstm = build_lstm((1, input_dim))
    hist_lstm = lstm.fit(X_train_rnn, y_train_seq, epochs=10, batch_size=32, verbose=1)
    lstm.save("model/lstm.h5")
    pd.DataFrame(hist_lstm.history).to_csv("model/lstm_loss.csv", index=False)
    lstm_pred = lstm.predict(X_test_rnn, verbose=0)
    lstm_rmse, lstm_mae, lstm_nrmse = evaluate_model(y_test_seq, lstm_pred)
    lstm_accuracy = display_predictions("LSTM", y_test_seq, lstm_pred, scaler)  
    results.append(["LSTM", lstm_rmse, lstm_mae, lstm_nrmse, lstm_accuracy])

    # GRU
    print("\nTraining GRU...")
    gru = build_gru((1, input_dim))
    hist_gru = gru.fit(X_train_rnn, y_train_seq, epochs=10, batch_size=32, verbose=1)
    gru.save("model/gru.h5")
    pd.DataFrame(hist_gru.history).to_csv("model/gru_loss.csv", index=False)
    gru_pred = gru.predict(X_test_rnn, verbose=0)
    gru_rmse, gru_mae, gru_nrmse = evaluate_model(y_test_seq, gru_pred)
    gru_accuracy = display_predictions("GRU", y_test_seq, gru_pred, scaler)     # add this
    results.append(["GRU", gru_rmse, gru_mae, gru_nrmse, gru_accuracy])

    # DT
    print("\nTraining DT...")
    dt = train_dt(X_train, y_train_seq)
    dt_pred = dt.predict(X_test)   
    dt_rmse, dt_mae, dt_nrmse = evaluate_model(y_test_seq, dt_pred)
    dt_accuracy = display_predictions("DT", y_test_seq, dt_pred, scaler)        # add this
    results.append(["DT", dt_rmse, dt_mae, dt_nrmse, dt_accuracy])

    # SAVE
    results_df = pd.DataFrame(results, columns=["model", "rmse", "mae", "nrmse", "accuracy"])
    results_df.to_csv("model/results.csv", index=False)
    print("\n==============================")
    print("FINAL RESULTS SAVED: model/results.csv")
    print("==============================")
    print(results_df)


# =========================
# RUN
# =========================
if __name__ == "__main__":
    main()
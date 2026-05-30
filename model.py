import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from graph import build_graph
import math

# NODES LIST
NODES_LIST = [
    3120, 3122, 3126, 3180, 4030,
    4032, 4034, 4035, 4040, 4043
]

STEP = 12

# LOAD MODELS
def load_models():
    lstm = load_model("model/lstm.h5", compile=False)
    gru = load_model("model/gru.h5", compile=False)
    dt = joblib.load("model/decision_tree.pkl")

    y_scaler = joblib.load("model/y_scaler.pkl")
    scaler_X = joblib.load("model/scaler_X.pkl")

    return lstm, gru, dt, y_scaler, scaler_X


# load historical scats traffic flow data, use latest 12 traffic values as prediction input
def load_traffic_histories():

    df = pd.read_csv("data/train.csv")

    traffic_data = {}

    for scats in df["SCATS Number"].unique():

        scats_df = df[df["SCATS Number"] == scats]

        flows = scats_df["flow_9to10"].tolist()

        # latest 12 history values
        history = flows[-STEP:]

        # padding
        history = [0] * (STEP - len(history)) + history

        traffic_data[scats] = history

    return traffic_data


# FLOW → SPEED CONVERSION
# Convert predicted traffic flow into vehicle speed
# Higher traffic flow = lower speed due to congestion
def flow_to_speed(flow):

    if flow <= 351:
        return 60.0

    a = 1.4648375
    b = -93.75
    c = flow

    discriminant = (b ** 2) - (4 * a * c)

    if discriminant < 0:
        return 20.0

    sqrt_disc = math.sqrt(discriminant)

    speed1 = (-b + sqrt_disc) / (2 * a)
    speed2 = (-b - sqrt_disc) / (2 * a)

    speed = max(speed1, speed2)

    speed = min(speed, 60.0)

    return speed


# PREDICTION HELPERS
# LSTM and GRU use sequential historical traffic data 
def predict_rnn(model, data):

    data = np.array(data, dtype=np.float32)

    expected_dim = model.input_shape[-1]

    # safety
    if len(data) > expected_dim:
        data = data[:expected_dim]

    elif len(data) < expected_dim:
        data = np.pad(data, (0, expected_dim - len(data)))

    timesteps = model.input_shape[1]
    features = model.input_shape[2]

    needed = timesteps * features

    if len(data) < needed:
        data = np.pad(
            data,
            (0, needed-len(data))
        )
    else:
        data = data[:needed]

    x = data.reshape(
        1,
        timesteps,
        features
    )

    prediction = model.predict(
        x,
        verbose=0
    )

    return float(prediction[0][0])

#Decision Tree uses engineered traffic features
def predict_tree(model, data):

    data = np.array(data, dtype=np.float32).reshape(1, -1)

    prediction = model.predict(data)

    return float(prediction[0])


# Build graph with ML-predicted travel times

def build_dynamic_graph(
    model,
    model_type,
    y_scaler=None,
    scaler_X=None
):

    G = build_graph()

    traffic_data = load_traffic_histories()

    # load once only
    if y_scaler is None:
        y_scaler = joblib.load(
            "model/y_scaler.pkl"
        )

    if scaler_X is None:
        scaler_X = joblib.load("model/scaler_X.pkl")

    # fixed date/time features
    hour = 9
    day_of_week = 1
    day = 1
    month = 10

    for u, v in G.edges():

        # get location history for v, use zeros if no history available, for each SCATS node
        history = traffic_data.get(v, [0] * STEP)

        history = history[-STEP:]

        history = [0] * (STEP - len(history)) + history

        # Encode SCATS location into machine learning features 1,0
        loc_onehot = [
            1 if n == v else 0
            for n in NODES_LIST
        ]

        # DT FEATURES
        # Create engineered features: previous flow, average flow, std deviation, time, location
        flow_t1 = history[-1]
        flow_t2 = history[-2]

        flow_mean_3 = np.mean(history[-3:])
        flow_std_3 = np.std(history[-3:])

        dt_features = [
            hour,
            day_of_week,
            day,
            month,
            flow_t1,
            flow_t2,
            flow_mean_3,
            flow_std_3
        ] + loc_onehot

        # RNN FEATURES - Combine traffic history + engineered features for LSTM/GRU
        rnn_features = history + dt_features

        # PREDICTION
        if model_type == "DT":

            dt_features_scaled = scaler_X.transform(
                np.array(dt_features).reshape(1, -1)
            )[0]

            predicted_flow = predict_tree(
                model,
                dt_features_scaled
            )

        else:

            predicted_flow = predict_rnn(
                model,
                rnn_features
            )

        # Scale back - Convert scaled prediction back into actual traffic flow
        predicted_flow = y_scaler.inverse_transform(
            [[predicted_flow]]
        )[0][0]

        # FLOW → SPEED → TIME
        # Convert predicted flow into speed using traffic equation.
        speed = flow_to_speed(predicted_flow)

        distance = G[u][v]["distance"]
        # Calculate travel time in minutes and add fixed intersection delay.
        travel_time = (distance / speed) * 60

        intersection_delay = 0.5  # 30 seconds = 0.5m

        travel_time += intersection_delay

        # Store travel time for display and for the search code.
        G[u][v]["travel_time"] = round(travel_time, 4)
        G[u][v]["weight"] = G[u][v]["travel_time"]

        print(
            f"{u} -> {v} | "
            f"Flow={predicted_flow:.2f} | "
            f"Travel Time={G[u][v]['travel_time']:.4f} min"
        )

    return G

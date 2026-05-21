import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from graph import build_graph
import math

# =========================
# NODES LIST
# =========================
NODES_LIST = [3120, 3122, 3126, 3180, 4030, 4032, 4034, 4035, 4040, 4043]

# =========================
# LOAD MODELS
# =========================
def load_models():
    lstm = load_model("model/lstm.h5", compile=False)
    gru = load_model("model/gru.h5", compile=False)
    dt = joblib.load("model/decision_tree.pkl")

    return lstm, gru, dt


# =========================
# LOAD REAL TRAFFIC HISTORY
# =========================
def load_traffic_histories():

    df = pd.read_csv("data/train.csv")

    traffic_data = {}

    scats_list = df["SCATS Number"].unique()

    for scats in scats_list:

        scats_df = df[df["SCATS Number"] == scats]

        flows = scats_df["flow_9to10"].tolist()

        # use latest 12 flows
        history = flows[-12:]

        # safety check
        if len(history) < 12:
            continue

        traffic_data[scats] = history

    return traffic_data


# =========================
# FLOW → SPEED CONVERSION
# Based on assignment PDF
# flow = -1.4648375(speed²) + 93.75(speed)
# =========================
def flow_to_speed(flow):

    # speed limit cap
    if flow <= 351:
        return 60.0

    a = 1.4648375
    b = -93.75
    c = flow

    discriminant = (b ** 2) - (4 * a * c)

    # safety check
    if discriminant < 0:
        return 20.0

    sqrt_disc = math.sqrt(discriminant)

    speed1 = (-b + sqrt_disc) / (2 * a)
    speed2 = (-b - sqrt_disc) / (2 * a)

    # under-capacity road = higher speed
    speed = max(speed1, speed2)

    # do not exceed speed limit
    speed = min(speed, 60.0)

    return speed



# =========================
# PREDICTION HELPERS
# =========================
def predict_rnn(model, data):
    # data is now the full feature vector (past_flow + meta), not just past flow
    x = np.array(data).reshape(1, 1, len(data))
    return float(model.predict(x, verbose=0)[0][0])


def predict_tree(model, data):

    prediction = model.predict(
        np.array(data).reshape(1, -1)
    )

    return float(prediction[0])


# =========================
# BUILD DYNAMIC GRAPH
# =========================
def build_dynamic_graph(model, model_type):
    G = build_graph()
    traffic_data = load_traffic_histories()

    # Meta features: hour=9, day_of_week=1 (Tuesday as placeholder), day=1, month=10
    # Ideally pass real date from GUI — for now use fixed October weekday
    meta_features = [9, 1, 1, 10]  # hour, day_of_week, day, month
    # + one-hot location (10 SCATS nodes) — all zeros except current node
    num_locations = len(NODES_LIST)  # import or define this

    for u, v in G.edges():
        history = traffic_data.get(v)
        if history is None:
            history = [100] * 12

        # Build location one-hot for node v
        loc_onehot = [1 if n == v else 0 for n in NODES_LIST]
        combined = history + meta_features + loc_onehot

        if model_type == "DT":
            predicted_flow = predict_tree(model, combined)
        else:
            predicted_flow = predict_rnn(model, combined)

        predicted_flow = max(1, min(predicted_flow * 1000, 2000))
        speed = flow_to_speed(predicted_flow)
        travel_time = (1.0 / speed) * 60 + 0.5
        G[u][v]["weight"] = round(travel_time, 4)

    return G
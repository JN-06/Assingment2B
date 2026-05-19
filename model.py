import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from graph import build_graph
import math


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

    x = np.array(data).reshape(1, len(data), 1)

    prediction = model.predict(x, verbose=0)

    return float(prediction[0][0])


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

    # REAL histories
    traffic_data = load_traffic_histories()

    for u, v in G.edges():

        # get destination node history
        history = traffic_data.get(v)

        # fallback safety
        if history is None:
            history = [100] * 12

        # =========================
        # MODEL PREDICTION
        # =========================
        if model_type == "DT":
            predicted_flow = predict_tree(model, history)
        else:
            predicted_flow = predict_rnn(model, history)


        # =========================
        # SAFETY CLAMP (ADD THIS)
        # =========================
        predicted_flow = max(0, predicted_flow)
        predicted_flow = min(predicted_flow, 2000)
        
        # =========================
        # SCALE BACK FLOW
        # (because train.py used MinMaxScaler)
        # =========================
        predicted_flow = predicted_flow * 1000

        # safety check
        predicted_flow = max(predicted_flow, 1)

        # =========================
        # FLOW → SPEED
        # =========================
        speed = flow_to_speed(predicted_flow)

        # =========================
        # TRAVEL TIME ESTIMATION
        # =========================

        # simplified distance assumption
        distance = 1.0  # km

        # time = distance / speed
        # convert hours → minutes
        travel_time = (distance / speed) * 60

        # add 30-second intersection delay
        travel_time += 0.5

        # =========================
        # STORE EDGE WEIGHT
        # =========================
        G[u][v]["weight"] = round(travel_time, 4)

    return G
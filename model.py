import numpy as np
import joblib
from tensorflow.keras.models import load_model
from graph import build_graph


# =========================
# LOAD MODELS
# =========================
def load_models():
    lstm = load_model("model/lstm.h5", compile=False)
    gru = load_model("model/gru.h5", compile=False)
    dt = joblib.load("model/decision_tree.pkl")
    return lstm, gru, dt


# =========================
# HELPERS
# =========================
def predict_rnn(model, data):
    x = np.array(data).reshape(1, len(data), 1)
    return float(model.predict(x, verbose=0)[0][0])


def predict_tree(model, data):
    return float(model.predict(np.array(data).reshape(1, -1))[0])


# =========================
# BUILD GRAPH FOR EACH MODEL
# =========================
def build_dynamic_graph(model, model_type):
    G = build_graph()

    history = [100,120,130,140,150,160,170,180,190,200,210,220]

    for u, v in G.edges():

        if model_type == "DT":
            cost = predict_tree(model, history)
        else:
            cost = predict_rnn(model, history)

        G[u][v]["weight"] = cost

    return G
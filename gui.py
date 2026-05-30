import tkinter as tk
from tkinter import ttk, messagebox

import networkx as nx
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D

import pandas as pd

from graph import build_graph
from model import load_models, build_dynamic_graph
from search import ida_star
from data import load_data
from model import NODES_LIST

import time

# =========================
# LOAD MODELS & DATA
# =========================
models = load_models()

lstm_model = models[0]
gru_model  = models[1]
dt_model   = models[2]
y_scaler   = models[3]
scaler_X   = models[4]

base_graph = build_graph()

# =========================
# RAW SCATS DATA
# =========================
raw_df = pd.read_excel(
    "Scats Data October 2006.xls",
    sheet_name="Data",
    engine="xlrd",
    header=1
)

raw_df.columns = raw_df.columns.str.strip()

df = load_data("Scats Data October 2006.xls")

df["SCATS Number"] = df["SCATS Number"].astype(int)

# =========================
# FIXED TIME
# 9AM - 10AM
# =========================
TIME_COLUMNS = ["V36", "V37", "V38", "V39"]

anim_path = []

current_graph = base_graph

current_model_type = "LSTM"

def safe_graph_fix(G):

    for u, v in G.edges():

        base_edge = base_graph.get_edge_data(u, v)

        # Preserve ML-predicted travel time.
        travel_time = float(
            G[u][v].get("travel_time", G[u][v].get("weight", 1.0)) or 1.0
        )
        G[u][v]["travel_time"] = travel_time
        G[u][v]["weight"] = travel_time

        # keep distance only
        if base_edge:
            G[u][v]["distance"] = float(base_edge.get("distance", 1.0))
        else:
            G[u][v]["distance"] = 1.0

    return G


def calculate_travel_time(G, path):

    total_time = 0.0

    for u, v in zip(path, path[1:]):

        edge = G.get_edge_data(u, v, {})

        total_time += float(
            edge.get("travel_time", edge.get("weight", 0.0)) or 0.0
        )

    return total_time


# =========================
# GET FIXED TIME COLUMNS
# =========================
def get_time_columns():

    return [
        c for c in TIME_COLUMNS
        if c in raw_df.columns
    ]


# =========================
# GET NODE FLOW
# =========================
def get_node_flow(scats_id):

    cols = get_time_columns()

    rows = raw_df[
        raw_df["SCATS Number"] == scats_id
    ]

    if rows.empty:
        return 0

    total = rows[cols].sum(axis=1).mean()

    return int(total)


# =========================
# DRAW GRAPH
# =========================
def draw_graph(G, path=None, highlight_index=-1, start_node=None, goal_node=None):

    ax.clear()

    pos = nx.get_node_attributes(G, "pos")

    ax.set_facecolor("#f0f4f8")

    # =========================
    # NORMAL EDGES
    # =========================
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color="gray",
        alpha=0.5,
        width=1.5,
        arrows=True
    )

    # =========================
    # NODES
    # =========================
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=2000,
        node_color="#f0f4f8",
        edgecolors="black",
        linewidths=1.5
    )

    # =========================
    # NODE LABELS
    # =========================
    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        font_size=9,
        font_weight="bold"
    )

    # =========================
    # START & GOAL HIGHLIGHT
    # =========================
    special_nodes = []

    if start_node is not None:
        special_nodes.append(start_node)

    if goal_node is not None:
        special_nodes.append(goal_node)

    if start_node is not None:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[start_node],
            ax=ax,
            node_color="#ffcc00",   # START = yellow
            node_size=2000
        )

    if goal_node is not None:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[goal_node],
            ax=ax,
            node_color="#ff3333",   # GOAL = red
            node_size=2000
        )

    # =========================
    # EDGE LABELS (COMBINED - FIXED SIZE)
    # =========================
    for (u, v, data) in G.edges(data=True):

        x = (pos[u][0] + pos[v][0]) / 2
        y = (pos[u][1] + pos[v][1]) / 2

        distance = data.get("distance", None)
        travel_time = data.get("travel_time", data.get("weight", None))

        box_style = dict(
            facecolor="white",
            edgecolor="none",
            boxstyle="round,pad=0.25"
        )

        # =========================
        # TOP → DISTANCE (RED)
        # =========================
        if distance is not None:
            ax.text(
                x,
                y + 0.04,
                f"D:{distance:>6.2f}",   # fixed width formatting
                fontsize=7,
                color="#ff0000",
                ha="center",
                va="center",
                family="monospace",      # IMPORTANT FIX
                bbox=box_style
            )

        # =========================
        # BOTTOM → WEIGHT (ORANGE)
        # =========================
        if travel_time is not None:
            ax.text(
                x,
                y - 0.04,
                f"T:{travel_time:>6.2f}", # fixed width formatting
                fontsize=7,
                color="#ff9500",
                ha="center",
                va="center",
                family="monospace",      # IMPORTANT FIX
                bbox=box_style
            )

    # =========================
    # PATH HIGHLIGHT
    # =========================
    if path and len(path) > 1:

        edges = list(
            zip(path, path[1:])
        )

        # ORANGE PATH
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edges,
            ax=ax,
            edge_color="#ff9500",
            width=1.5,
            arrows=True
        )

        # GREEN VISITED NODES
        if highlight_index >= 0:

            visited = path[:highlight_index + 1]

            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=visited,
                ax=ax,
                node_color="#00cc66",
                node_size=750
            )

    # =========================
    # LEGEND
    # =========================
    legend_elements = [

        Line2D([0], [0],
            color="gray",
            lw=2,
            label="Edge (Road connection)"),

        Line2D([0], [0],
            color="#ff9500",
            lw=0,
            label="Distance (Base road length)"),

        Line2D([0], [0],
            color="#ff0000",
            lw=0,
            label="Travel Time (Predicted)"),

        Line2D([0], [0],
            marker='o',
            color='w',
            markerfacecolor="#00cc66",
            markersize=10,
            label="Visited node"),

        Line2D([0], [0],
            marker='o',
            color='w',
            markerfacecolor="#f0f4f8",
            markersize=10,
            label="Unvisited node"),

        Line2D([0], [0],
            marker='o',
            color='w',
            markerfacecolor="#ffcc00",
            markersize=10,
            label="Start node"),

        Line2D([0], [0],
            marker='o',
            color='w',
            markerfacecolor="#ff3333",
            markersize=10,
            label="Goal node")
    ]

    legend = ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=9,
        frameon=True
    )

    # =========================
    # COLOR LEGEND TEXT
    # =========================
    for text in legend.get_texts():
        label = text.get_text()

        if "Distance" in label:
            text.set_color("red")

        elif "Time" in label:
            text.set_color("orange")

    # =========================
    # TITLE
    # =========================
    ax.set_title(
        f"SCATS Traffic Flow Prediction ({current_model_type})",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_axis_off()

    canvas.draw()


# =========================
# ANIMATION
# =========================
def animate(i):

    if i >= len(anim_path):
        return

    draw_graph(current_graph, anim_path, i, start_global, goal_global)

    root.after(
        700,
        animate,
        i + 1
    )

# =========================
# RUN MODEL
# =========================
def run_model():

    global anim_path, current_graph, current_model_type
    global start_global, goal_global

    try:
        start = int(start_var.get())
        goal = int(goal_var.get())
    except:
        messagebox.showerror("Input Error", "Please select start and goal nodes")
        return

    if start == goal:
        messagebox.showerror("Input Error", "Start and goal cannot be same")
        return

    model_type = model_var.get()
    current_model_type = model_type

    start_global = start
    goal_global = goal

    model = {
        "LSTM": lstm_model,
        "GRU": gru_model,
        "DT": dt_model
    }[model_type]

    dynamic_graph = build_dynamic_graph(
        model,
        model_type,
        y_scaler,
        scaler_X
    )

    # =========================
    # 🔥 FIX APPLIED (NO LOGIC REMOVED)
    # =========================
    current_graph = safe_graph_fix(dynamic_graph)

    coords = nx.get_node_attributes(current_graph, "pos")

    start_time = time.time()

    path = ida_star(
        current_graph,
        coords,
        start,
        [goal]
    )

    execution_time = time.time() - start_time

    if not path:
        messagebox.showerror("No Path", "No path found")
        return

    anim_path = path

    estimated_travel_time = calculate_travel_time(current_graph, path)

    total_flow = 0
    node_flows = []

    for node in path:

        flow = get_node_flow(node)

        node_flows.append(f"{node}: {flow}")

        total_flow += flow

    result_text.set(
        f"==============================\n"
        f"MODEL : {model_type}\n"
        f"ALGORITHM : IDA*\n"
        f"TIME : 9AM - 10AM\n"
        f"==============================\n"
        f"PATH : {' -> '.join(map(str, path))}\n\n"
        f"NODE FLOWS:\n"
        f"{chr(10).join(node_flows)}\n\n"
        f"TOTAL FLOW : {total_flow}\n"
        f"EST. TRAVEL TIME : {estimated_travel_time:.2f} min\n"
        f"NODES VISITED : {len(path)}\n"
        f"EXECUTION TIME : {execution_time*1000:.3f} ms\n"
    )

    animate(0)


def predict_flow_for_node(model, model_type, node):

    import numpy as np
    from model import load_traffic_histories, NODES_LIST

    traffic_data = load_traffic_histories()

    history = traffic_data.get(node, [0] * 12)
    history = history[-12:]
    history = [0] * (12 - len(history)) + history

    # =========================
    # META FEATURES (MATCH TRAINING)
    # =========================
    flow_t1 = history[-1]
    flow_t2 = history[-2]
    flow_mean_3 = np.mean(history[-3:])
    flow_std_3 = np.std(history[-3:])

    meta_features = [
        9,   # hour
        1,   # day_of_week
        1,   # day
        10,  # month
        flow_t1,
        flow_t2,
        flow_mean_3,
        flow_std_3
    ]

    loc_onehot = [1 if n == node else 0 for n in NODES_LIST]

    dt_features = meta_features + loc_onehot

    # =========================
    # DT MODEL
    # =========================
    if model_type == "DT":

        x = np.array(dt_features).reshape(1, -1)

        x = scaler_X.transform(x)

        return float(model.predict(x)[0])

    # =========================
    # RNN MODEL (LSTM / GRU)
    # =========================
    x = np.array(
    history + dt_features,
    dtype=np.float32
    )

    # get model dimensions
    timesteps = model.input_shape[1]
    features = model.input_shape[2]

    needed = timesteps * features

    # padding / trimming
    if len(x) < needed:
        x = np.pad(
            x,
            (0, needed - len(x))
        )
    else:
        x = x[:needed]

    # CORRECT SHAPE
    x = x.reshape(
        1,
        timesteps,
        features
    )

    return float(
        model.predict(
            x,
            verbose=0
        )[0][0]
    )

# =========================
# COMPARE MODELS
# =========================
def compare_all():

    try:
        start = int(start_var.get())
        goal = int(goal_var.get())
    except:
        messagebox.showerror("Error", "Select start and goal")
        return

    results = []

    for name, model, mtype in [
        ("LSTM", lstm_model, "LSTM"),
        ("GRU", gru_model, "GRU"),
        ("DT", dt_model, "DT")
    ]:

        try:
            G = build_dynamic_graph(model, mtype, y_scaler, scaler_X)
            G = safe_graph_fix(G)

            coords = nx.get_node_attributes(G, "pos")
            path = ida_star(G, coords, start, [goal])

            if not path:
                continue

            estimated_travel_time = calculate_travel_time(G, path)

            actual_total = 0
            predicted_total = 0

            for node in path:

                actual = get_node_flow(node)

                actual_total += actual

                # convert scaled value back to real traffic flow
                pred_raw = predict_flow_for_node(model, mtype, node)

                pred_final = y_scaler.inverse_transform([[pred_raw]])[0][0]

                print(
                    f"{name} | Node {node} | "
                    f"Actual={actual} | "
                    f"Predicted={pred_final:.2f}"
                )

                predicted_total += pred_final

            error = abs(
                actual_total - predicted_total
            )

            results.append((name, len(path),
                    actual_total, predicted_total, error,
                    estimated_travel_time, path))

        except Exception as e:
            print(f"{name} failed:", e)
            continue

    if not results:
        messagebox.showerror("Error", "No paths found")
        return

    win = tk.Toplevel(root)
    win.title("Model Comparison")

    tree = ttk.Treeview(
        win,
        columns=("Model", "Nodes", "Actual", "Predicted", "Error", "Travel Time", "Path"),
        show="headings"
    )

    for c in ("Model", "Nodes", "Actual", "Predicted", "Error", "Travel Time", "Path"):
        tree.heading(c, text=c)

    best = max(results, key=lambda x: x[5])

    for r in results:
        tag = ("best",) if r[0] == best[0] else ()

        tree.insert("", tk.END, values=(
            r[0],
            r[1],
            int(r[2]),
            int(r[3]),
            int(r[4]),
            f"{r[5]:.2f} min",
            " -> ".join(map(str, r[6]))
        ), tags=tag)

    tree.tag_configure("best", background="lightgreen")
    tree.pack(fill=tk.BOTH, expand=True)


# =========================
# RESET
# =========================
def reset():

    global current_graph
    global anim_path

    current_graph = base_graph

    anim_path = []

    result_text.set("")

    draw_graph(base_graph, start_node=None, goal_node=None)


# =========================
# GUI
# =========================
root = tk.Tk()

root.title(
    "SCATS Traffic Flow Prediction System"
)

root.geometry("1350x820")

# =========================
# LEFT PANEL
# =========================
left = tk.Frame(
    root,
    padx=10,
    pady=10
)

left.pack(
    side=tk.LEFT,
    fill=tk.Y
)

title = tk.Label(
    left,
    text="Traffic Flow Prediction",
    font=("Arial", 16, "bold")
)

title.pack(pady=10)

# =========================
# START
# =========================
tk.Label(
    left,
    text="Start SCATS"
).pack()

start_var = ttk.Combobox(
    left,
    values=sorted(base_graph.nodes()),
    width=25
)

start_var.pack(pady=5)

# =========================
# GOAL
# =========================
tk.Label(
    left,
    text="Goal SCATS"
).pack()

goal_var = ttk.Combobox(
    left,
    values=sorted(base_graph.nodes()),
    width=25
)

goal_var.pack(pady=5)

# =========================
# MODEL
# =========================
tk.Label(
    left,
    text="Prediction Model"
).pack()

model_var = ttk.Combobox(
    left,
    values=[
        "LSTM",
        "GRU",
        "DT"
    ],
    width=25
)

model_var.set("LSTM")

model_var.pack(pady=5)

# =========================
# BUTTONS
# =========================
tk.Button(
    left,
    text="Run Model",
    command=run_model,
    bg="#4CAF50",
    fg="white",
    width=25
).pack(pady=10)

tk.Button(
    left,
    text="Compare Models",
    command=compare_all,
    bg="#2196F3",
    fg="white",
    width=25
).pack(pady=5)

tk.Button(
    left,
    text="Reset",
    command=reset,
    bg="#f44336",
    fg="white",
    width=25
).pack(pady=5)

# =========================
# RESULT DISPLAY
# =========================
result_text = tk.StringVar()

result_label = tk.Label(
    left,
    textvariable=result_text,
    justify="left",
    anchor="w",
    font=("Consolas", 10),
    bg="#f4f4f4",
    relief="solid",
    padx=10,
    pady=10
)

result_label.pack(
    fill=tk.BOTH,
    expand=True,
    pady=15
)

# =========================
# RIGHT PANEL
# =========================
right = tk.Frame(root)

right.pack(
    side=tk.RIGHT,
    fill=tk.BOTH,
    expand=True
)

fig, ax = plt.subplots(
    figsize=(10, 7)
)

canvas = FigureCanvasTkAgg(
    fig,
    master=right
)

canvas.get_tk_widget().pack(
    fill=tk.BOTH,
    expand=True
)

draw_graph(base_graph)

# =========================
# RUN GUI
# =========================
root.mainloop()

import tkinter as tk
from tkinter import ttk, messagebox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import datetime

from graph import build_graph
from model import load_models, build_dynamic_graph
from search import ida_star
from data import load_data

# =========================
# LOAD MODELS & DATA
# =========================
lstm_model, gru_model, dt_model = load_models()
base_graph = build_graph()

df = load_data("Scats Data October 2006.xls")
print(df.columns.tolist())
df["SCATS Number"] = df["SCATS Number"].astype(int)

anim_path = []
current_graph = base_graph
current_model_type = "LSTM"

# =========================
# SCATS CONFIG
# =========================
SCATS_START_HOUR = 9
SCATS_END_HOUR = 22

# =========================
# COST CALCULATION
# =========================
def calculate_cost(G, path):
    if not path or len(path) < 2:
        return 0
    cost = 0
    for i in range(len(path) - 1):
        cost += G[path[i]][path[i + 1]].get("weight", 1.0)
    return cost


# =========================
# DRAW GRAPH
# =========================
def draw_graph(G, path=None, highlight_index=-1):
    ax.clear()
    pos = nx.get_node_attributes(G, "pos")
    ax.set_facecolor("#f0f4f8")

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray", alpha=0.4, width=1.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=600,
                            node_color="#4da3ff",
                            edgecolors="black", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)

    if path and len(path) > 1:
        edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=edges, ax=ax,
                               edge_color="#ff9500", width=4)

        if highlight_index >= 0:
            visited = path[:highlight_index + 1]
            nx.draw_networkx_nodes(G, pos, nodelist=visited, ax=ax,
                                   node_color="#00cc66", node_size=700)

    ax.set_title("SCATS Route System", fontsize=14)
    ax.set_axis_off()
    canvas.draw()


# =========================
# ANIMATION
# =========================
def animate(i):
    if i >= len(anim_path):
        return
    draw_graph(current_graph, anim_path, i)
    root.after(600, animate, i + 1)


# =========================
# RUN MODEL
# =========================
def run_model():
    global anim_path, current_graph, current_model_type

    try:
        start = int(start_var.get())
        goal = int(goal_var.get())
    except:
        messagebox.showerror("Input Error", "Select start and goal")
        return

    if start == goal:
        messagebox.showerror("Input Error", "Start and goal cannot be same")
        return

    model_type = model_var.get()
    algorithm = algo_var.get()

    selected_hour = int(hour_dropdown.get())

    # =========================
    # FIXED TIME COLUMNS (CORRECT)
    # =========================
    def get_time_columns(hour):
        """
        SCATS range: V36 → V91 (09:00–22:45)
        """

        if hour < SCATS_START_HOUR or hour > SCATS_END_HOUR:
            return []

        start_index = hour * 4
        cols = [f"V{start_index + i}" for i in range(4)]

        return [c for c in cols if c in df.columns]

    # =========================
    # MODEL
    # =========================
    model = {
        "LSTM": lstm_model,
        "GRU": gru_model,
        "DT": dt_model
    }[model_type]

    current_graph = build_dynamic_graph(model, model_type)

    # =========================
    # PATH SEARCH
    # =========================
    if algorithm == "IDA*":
        path = ida_star(current_graph, start, goal)
    else:
        path = simple_dfs(current_graph, start, goal)

    if not path:
        messagebox.showerror("No Path", "No path found")
        return

    anim_path = path
    cost = calculate_cost(current_graph, path)

    # =========================
    # TOTAL FLOW (FIXED)
    # =========================
    # =========================
    # SCATS FLOW COLUMNS FIX
    # =========================
    selected_hour = int(hour_dropdown.get())

    if selected_hour < 9 or selected_hour > 22:
        messagebox.showerror(
            "Invalid Time",
            "Only SCATS range allowed: 09:00–22:45 (V36–V91)"
        )
        return


    # def get_time_columns(hour):
    #     start_index = hour * 4
    #     cols = [f"V{start_index + i}" for i in range(4)]
    #     return [c for c in cols if c in df.columns]


    cols = get_time_columns(selected_hour)
    total_flow = df[cols].sum(axis=1).sum() if cols else 0

    result_text.set(
        f"Model: {model_type}\n"
        f"Algorithm: {algorithm}\n"
        f"Hour: {selected_hour}:00\n"
        f"Total Flow: {total_flow}\n"
        f"Path: {' → '.join(map(str, path))}\n"
        f"Cost: {cost:.4f}\n"
        f"Nodes: {len(path)}"
    )

    animate(0)


# =========================
# DFS
# =========================
def simple_dfs(G, start, goal):
    stack = [(start, [start])]
    visited = set()

    while stack:
        node, path = stack.pop()

        if node == goal:
            return path

        if node in visited:
            continue

        visited.add(node)

        for nxt in G[node]:
            if nxt not in path:
                stack.append((nxt, path + [nxt]))

    return None


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
        G = build_dynamic_graph(model, mtype)
        path = ida_star(G, start, goal)

        if path:
            cost = calculate_cost(G, path)
            results.append((name, cost, len(path), path))

    if not results:
        messagebox.showerror("Error", "No paths found")
        return

    win = tk.Toplevel(root)
    win.title("Model Comparison")

    tree = ttk.Treeview(win,
                        columns=("Model", "Cost", "Nodes", "Path"),
                        show="headings")

    for c in ("Model", "Cost", "Nodes", "Path"):
        tree.heading(c, text=c)

    tree.pack(fill=tk.BOTH, expand=True)

    best = min(results, key=lambda x: x[1])

    for r in results:
        tag = ("best",) if r[0] == best[0] else ()
        tree.insert("", tk.END,
                    values=(r[0], f"{r[1]:.4f}", r[2], "→".join(map(str, r[3]))),
                    tags=tag)

    tree.tag_configure("best", background="lightgreen")


# =========================
# RESET
# =========================
def reset():
    global current_graph, anim_path
    current_graph = base_graph
    anim_path = []
    result_text.set("")
    draw_graph(base_graph)


# =========================
# GUI
# =========================
root = tk.Tk()
root.title("SCATS Route System")
root.geometry("1250x780")

left = tk.Frame(root)
left.pack(side=tk.LEFT, fill=tk.Y)

tk.Label(left, text="Start").pack()
start_var = ttk.Combobox(left, values=sorted(base_graph.nodes()))
start_var.pack()

tk.Label(left, text="Goal").pack()
goal_var = ttk.Combobox(left, values=sorted(base_graph.nodes()))
goal_var.pack()

tk.Label(left, text="Model").pack()
model_var = ttk.Combobox(left, values=["LSTM", "GRU", "DT"])
model_var.set("LSTM")
model_var.pack()

tk.Label(left, text="Algorithm").pack()
algo_var = ttk.Combobox(left, values=["IDA*", "DFS"])
algo_var.set("IDA*")
algo_var.pack()

tk.Label(left, text="Hour (9-22)").pack()
hour_dropdown = ttk.Combobox(left, values=list(range(9, 23)))
hour_dropdown.current(0)
hour_dropdown.pack()

tk.Button(left, text="Run", command=run_model).pack()
tk.Button(left, text="Compare", command=compare_all).pack()
tk.Button(left, text="Reset", command=reset).pack()

result_text = tk.StringVar()
tk.Label(left, textvariable=result_text, justify="left").pack()

right = tk.Frame(root)
right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

fig, ax = plt.subplots(figsize=(8, 6))
canvas = FigureCanvasTkAgg(fig, master=right)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

draw_graph(base_graph)

root.mainloop()
import math

def heuristic(a, b):
    return 0


def dfs(path, g, threshold, goal, G):
    node = path[-1]
    f = g + heuristic(node, goal)

    if f > threshold:
        return f

    if node == goal:
        return -1

    min_cost = math.inf

    for neighbor in G[node]:
        if neighbor in path:
            continue

        cost = G[node][neighbor]["weight"]
        path.append(neighbor)

        t = dfs(path, g + cost, threshold, goal, G)

        if t == -1:
            return -1

        if t < min_cost:
            min_cost = t

        path.pop()

    return min_cost


def ida_star(G, start, goal):
    # simple DFS fallback (safe for assignment demo)

    stack = [(start, [start])]
    visited = set()

    while stack:
        node, path = stack.pop()

        if node == goal:
            return path

        if node in visited:
            continue

        visited.add(node)

        for neighbor in G[node]:
            stack.append((neighbor, path + [neighbor]))

    return None
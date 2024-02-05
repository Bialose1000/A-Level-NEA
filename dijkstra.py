import matplotlib.pyplot as plt
import numpy as np
import heapq

def completion_check(qmaze):
    start = qmaze.rat.get_position()
    end = qmaze.target
    shortest_path = dijkstra(qmaze.maze, start, end)
    if shortest_path:
        print("Shortest path found with Dijkstra's algorithm:", shortest_path)
        display_shortest_path(qmaze.maze, shortest_path)  # Display the shortest path
    else:
        print("No path found!")

def display_shortest_path(maze, shortest_path):
    nrows, ncols = maze.shape
    canvas = np.copy(maze)
    for r, c in shortest_path:
        canvas[r, c] = 0.5  # Mark the shortest path on the canvas

    plt.imshow(canvas, cmap='viridis_r', origin = 'upper', vmin=0, vmax=1)
    plt.show()

def dijkstra(maze, start, end):
    rows, cols = maze.shape
    dist = np.full((rows, cols), np.inf)
    dist[start] = 0
    pq = [(0, start)]
    prev = np.empty((rows, cols, 2), dtype=int)
    prev[start] = -1, -1

    while pq:
        d, (x, y) = heapq.heappop(pq)

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy

            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] == 1:
                next_dist = d + 1

                if next_dist < dist[nx, ny]:
                    dist[nx, ny] = next_dist
                    prev[nx, ny] = x, y
                    heapq.heappush(pq, (next_dist, (nx, ny)))

    path = []
    x, y = end
    while (x, y) != (-1, -1):
        path.append((x, y))
        x, y = prev[x, y]
    path.reverse()

    return path if dist[end] != np.inf else None



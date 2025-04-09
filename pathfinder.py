# import numpy as np
# import heapq

# def a_star(density_map, start, goal):
#     height, width = density_map.shape
#     visited = set()
#     heap = [(0, start)]
#     came_from = {}
#     cost_so_far = {start: 0}

#     def neighbors(pos):
#         x, y = pos
#         for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
#             nx, ny = x + dx, y + dy
#             if 0 <= nx < height and 0 <= ny < width:
#                 yield (nx, ny)

#     def heuristic(a, b):
#         return abs(a[0]-b[0]) + abs(a[1]-b[1])

#     while heap:
#         _, current = heapq.heappop(heap)

#         if current == goal:
#             break

#         if current in visited:
#             continue
#         visited.add(current)

#         for next_node in neighbors(current):
#             cost = density_map[next_node] + 1e-3
#             new_cost = cost_so_far[current] + cost
#             if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
#                 cost_so_far[next_node] = new_cost
#                 priority = new_cost + heuristic(goal, next_node)
#                 heapq.heappush(heap, (priority, next_node))
#                 came_from[next_node] = current

#     path = []
#     curr = goal
#     while curr != start:
#         path.append(curr)
#         curr = came_from.get(curr)
#         if curr is None:
#             return []
#     path.append(start)
#     path.reverse()
#     return path

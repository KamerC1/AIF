import networkx as nx

class HeldKarp:
    def __init__(self, world_graph, gold_positions, start):
        self.world_graph = world_graph
        self.gold_positions = gold_positions
        self.start = start
        distances = self.calculate_distances(world_graph, gold_positions + [start])
        self.dist = [[distances[i][j] for j in distances[i]] for i in distances]  # matrix of distances
        self.n = len(self.dist) - 1

    def calculate_distances(self, G, points):
        """
        Restituisce un dizionare dove per ogni punto calcola la distanza minima con tutti gli altri punti.
        """
        distances = {}
        for point in points:
            lengths = nx.single_source_dijkstra_path_length(G, point)
            distances[point] = {other: lengths[other] for other in points if other in lengths}
        return distances

    def solve_tsp(self):
        # This code is contributed by Serjeel Ranjan (https://www.geeksforgeeks.org/traveling-salesman-problem-tsp-in-python/)
        memo = [[-1] * (1 << (self.n + 1)) for _ in range(self.n + 1)]
        
        def fun(i, mask):
            if mask == ((1 << i) | 3):
                return self.dist[1][i]

            if memo[i][mask] != -1:
                return memo[i][mask]

            res = 10**9

            for j in range(1, self.n + 1):
                if (mask & (1 << j)) != 0 and j != i and j != 1:
                    res = min(res, fun(j, mask & (~(1 << i))) + self.dist[j][i])
            
            memo[i][mask] = res
            return res

        ans = 10**9
        for i in range(1, self.n + 1):
            ans = min(ans, fun(i, (1 << (self.n + 1)) - 1) + self.dist[i][1])
        
        return ans
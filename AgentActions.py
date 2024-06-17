import networkx as nx

class AgentActions:
    def __init__(self, world_graph: nx.Graph, starting_position: tuple):
        self.world_graph = world_graph
        self.starting_position = starting_position
        
    def get_agent_actions(self, best_route: list):
        """
        Genera le azioni dell'agente a partire dal percorso migliore.
        """
        
        agent_movements = []
        best_route = [self.starting_position] + best_route
        for i in range(1, len(best_route)):
            start_rout = best_route[i - 1]
            end_rout = best_route[i]

            shortest_path = self._find_shortest_path(start_rout, end_rout)
            if shortest_path:
                dPad_movements = self._get_dPad_movement_from_path(shortest_path)
                # Aggiungi i movimenti e l'azione di raccolta "C" alla lista delle azioni
                agent_movements += [dPad_movements + ["C"]]

        return agent_movements
    
    def _find_shortest_path(self, start: tuple, goal: tuple):
        """
            list: Il percorso più breve come lista di tuple.
        """
        try:
            path = nx.astar_path(
                self.world_graph, start, goal,
                heuristic=lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])
            )
            return path
        except nx.NetworkXNoPath:
            return None

    def _get_dPad_movement(self, start_position: tuple, end_position: tuple):
        """
        Converte una coppia di posizioni in un movimento del d-pad.
        """
        
        delta_x = end_position[0] - start_position[0]
        delta_y = end_position[1] - start_position[1]

        direction = {
            (-1, 1): "NE",
            (-1, -1): "NW",
            (1, 1): "SE",
            (1, -1): "SW",
            (1, 0): "S",
            (-1, 0): "N",
            (0, 1): "E",
            (0, -1): "W"
        }

        return direction.get((delta_x, delta_y), None)

    def _get_dPad_movement_from_path(self, path: list):
        """
        Converte un percorso in una serie di movimenti del d-pad.
        """
        
        movements = []
        for i in range(len(path) - 1):
            start_position = path[i]
            end_position = path[i + 1]
            movements.append(self._get_dPad_movement(start_position, end_position))

        return movements
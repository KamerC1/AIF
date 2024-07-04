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
                # Aggiungi l'azione di raccolta "C" alla lista delle azioni
                agent_movements += dPad_movements + ["C"]
        
        agent_movements.pop()

        return self._convert_actions_to_integer(agent_movements)
    
    def _find_shortest_path(self, start: tuple, goal: tuple):
        """
        Trova il percorso più breve tra start e goal utilizzando l'algoritmo A*.
        
        Args:
            start (tuple): La posizione di partenza.
            goal (tuple): La posizione di destinazione.
        
        Returns:
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
        
        Args:
            start_position (tuple): La posizione di partenza.
            end_position (tuple): La posizione di destinazione.
        
        Returns:
            str: La direzione del movimento del d-pad.
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
        
        Args:
            path (list): Il percorso come lista di posizioni.
        
        Returns:
            list: I movimenti del d-pad come lista di stringhe.
        """
        
        movements = []
        for i in range(len(path) - 1):
            start_position = path[i]
            end_position = path[i + 1]
            movements.append(self._get_dPad_movement(start_position, end_position))

        return movements
    
    def _convert_actions_to_integer(self, actions_array):
        """
        Converte un array di azioni in un array di interi secondo una mappa di traduzione.
        """
        
        translation = {
            'N': 0,
            'E': 1,
            'S': 2,
            'W': 3,
            'NE': 4,
            'SE': 5,
            'SW': 6,
            'NW': 7,
            'C': 49  # Raccoglie oggetto
        }
        
        int_actions = []
        
        for action in actions_array:
            int_actions.append(translation[action] if action in translation else None)
        
        return int_actions
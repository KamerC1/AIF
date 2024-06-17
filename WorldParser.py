import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import re
from typing import List, Tuple

class WorldParser:
    def __init__(self, des_file: str):
        self.des_file = des_file

    def extract_gold_positions(self) -> List[Tuple[int, int]]:
        """
        Returns: Lista di tuple con le coordinate dei punti d'oro.
        """
        
        gold_pattern = r'GOLD:1,\((\d+),(\d+)\)'
        gold_positions = []
        matches = re.findall(gold_pattern, self.des_file)

        for match in matches:
            x, y = map(int, match)
            gold_positions.append((x, y))
            
        gold_positions = [(b, a) for a, b in gold_positions]

        return gold_positions
    
    def extract_world_graph(self) -> nx.Graph:
        """
        Returns: Grafo rappresentante il mondo senza i muri.
        """
        
        map_layout = self._parse_map()
        walls_positions = self._extract_wall_positions(map_layout)
        
        G = nx.grid_2d_graph(len(map_layout), len(map_layout[0]))
        
        # Aggiungi connessioni diagonali
        for node in G.nodes():
            x, y = node
            # Aggiungi connessioni diagonali (nei quattro angoli)
            if (x+1, y+1) in G.nodes():
                G.add_edge(node, (x+1, y+1))
            if (x-1, y-1) in G.nodes():
                G.add_edge(node, (x-1, y-1))
            if (x+1, y-1) in G.nodes():
                G.add_edge(node, (x+1, y-1))
            if (x-1, y+1) in G.nodes():
                G.add_edge(node, (x-1, y+1))
        
        G.remove_nodes_from(walls_positions)
        
        return G
    
    def _parse_map(self) -> List[List[str]]:
        """
        Returns: Mappa rappresentata come lista di liste di caratteri.
        """
        
        start_map = self.des_file.index("MAP") + 3
        end_map = self.des_file.index("ENDMAP")
        map_lines = self.des_file[start_map:end_map].strip().split('\n')
        return [list(line.strip()) for line in map_lines]

    def _extract_wall_positions(self, map_layout: List[List[str]]) -> List[Tuple[int, int]]:
        """
        Returns: Lista di tuple con le coordinate dei muri.
        """
        
        map_layout = np.array(map_layout)
        walls_positions = list(zip(*np.where(map_layout == '|')))
        return walls_positions


    def plot_world_graph(self, world_graph: nx.Graph, gold_positions: List[Tuple[int, int]]):
        """
        Plotta il grafo del mondo con le posizioni degli oggetti e degli ostacoli.

        Args:
            world_graph (nx.Graph): Grafo del mondo senza muri.
            gold_positions (List[Tuple[int, int]]): Lista di tuple con le coordinate dei punti d'oro.
        """
        
        plt.figure(figsize=(10, 8))
        pos = {(x, y): (y, x) for x, y in world_graph.nodes()}
        nx.draw(world_graph, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=8, font_color='black')

        # Aggiungiamo le etichette degli oggetti (gold) in rosso
        for i, (x, y) in enumerate(gold_positions, start=1):
            plt.text(y, x, f"{i}\n\n", ha='center', va='center', color='red', fontweight='bold')

        plt.title('Grid Graph with Objects and Obstacles')
        plt.gca().invert_yaxis()  # Invertiamo l'asse y per avere l'orientazione corretta
        plt.show()

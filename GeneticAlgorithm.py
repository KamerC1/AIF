import random
import networkx as nx
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from typing import List, Tuple

class GeneticAlgorithm:
    def __init__(self, world_graph: nx.Graph, gold_positions: List[Tuple[int, int]], starting_position: Tuple[int, int]):
        """
        Inizializza l'ottimizzatore per raccogliere l'oro.

        Args:
            world_graph (nx.Graph): Grafo del mondo.
            gold_positions (List[Tuple[int, int]]): Posizioni degli oggetti d'oro.
            starting_position (Tuple[int, int], optional): Posizione di partenza. Default è (0, 0).
        """
        self.G = world_graph
        self.gold_positions = gold_positions
        self.starting_position = starting_position
        self.num_elements = len(gold_positions)
        
        self.toolbox = base.Toolbox()
        self.configure_toolbox()
        
    def a_star_distance(self, start: Tuple[int, int], goal: Tuple[int, int]) -> int:
        """
        Calcola la distanza di Manhattan tra due punti usando A*.
        """
        try:
            path = nx.astar_path(self.G, start, goal, heuristic=lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1]))
            return len(path) - 1
        except nx.NetworkXNoPath:
            return float('inf')

    def evaluate(self, individual: List[int]) -> Tuple[int]:
        """
        Calcola la lunghezza del percorso per un individuo.
        """
        
        #Distanza dal punto di partenza al primo oggetto
        distance = self.a_star_distance(self.starting_position, self.gold_positions[individual[0]])
        
        # Distanza tra gli oggetti
        for i in range(len(individual) - 1):
            distance += self.a_star_distance(self.gold_positions[individual[i]], self.gold_positions[individual[i + 1]])
        
        # Aggiungi la distanza dall'ultimo oggetto al punto di partenza
        distance += self.a_star_distance(self.gold_positions[individual[-1]], self.starting_position)
        
        return distance,

    def configure_toolbox(self):
        
        # Definizione del tipo di problema e fitness
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        # Registrazione delle funzioni genetiche
        self.toolbox.register("indices", random.sample, range(self.num_elements), self.num_elements)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.indices)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Funzioni di crossover e mutazione
        self.toolbox.register("mate", tools.cxOrdered)
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.02)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.evaluate)

    def optimize(self, population_size: int = 10, num_generations: int = 500, cxpb: float = 0.7, mutpb: float = 0.2):
        """
        Esegue l'algoritmo genetico per ottimizzare il percorso.
        """
        
        population = self.toolbox.population(n=population_size)
        
        algorithms.eaSimple(population, self.toolbox, cxpb, mutpb, num_generations, 
                            stats=tools.Statistics(lambda ind: ind.fitness.values),
                            halloffame=tools.HallOfFame(1),
                            verbose=False)
        
        best_individual = tools.selBest(population, 1)[0]
        best_route = [self.gold_positions[i] for i in best_individual]
        best_route = best_route + [self.starting_position]
        best_distance = self.evaluate(best_individual)[0]
        
        return best_route, best_distance

    def plot_world_graph(self, best_route: List[Tuple[int, int]]):
        """
        Plotta il grafo del mondo con il miglior percorso trovato.
        """
        
        plt.figure(figsize=(10, 8))
        pos = {(x, y): (y, x) for x, y in self.G.nodes()}
        nx.draw(self.G, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=8, font_color='black')

        # Aggiungiamo il percorso dal punto di partenza al primo oggetto
        start_to_first_gold = [self.starting_position, best_route[0]]
        nx.draw_networkx_edges(self.G, pos, edgelist=[start_to_first_gold], edge_color='blue', width=2)

        # Add labels for gold positions in red, based on order of visit
        for idx, gold in enumerate(best_route, start=1):
            if gold in self.gold_positions:
                plt.text(gold[1], gold[0], str(idx) + "\n\n", ha='center', va='center', color='red', fontweight='bold')

        path_edges = list(zip(best_route[:-1], best_route[1:]))
        nx.draw_networkx_edges(self.G, pos, edgelist=path_edges, edge_color='green', width=2)
        
        plt.title('Grid Graph with Gold Positions')
        plt.gca().invert_yaxis()
        plt.show()

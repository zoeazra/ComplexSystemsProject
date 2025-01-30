import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import sys
import time
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from file_system.file_sys import *

# A static without time network model
def static_network_model(N, probabilities):
    """
    A static network model with N nodes and collision probability list.
    """
    for p in probabilities:
        G = nx.random_graphs.erdos_renyi_graph(N, p)
        largest_cc2 = max(nx.connected_components(G), key=len)
        avg_degree2 = np.mean([d for _, d in G.degree()])
        gc_size2 = len(largest_cc2)
        write(0, 0, 0, 0, 0, N, p, gc_size2, avg_degree2, "../../results", "static", "static")

def sample_collisions(nodes, P):
    """
    fast generation of collision pairs, avoiding explicit iteration over all pairs of nodes
    """
    N = len(nodes)
    num_possible_edges = N * (N - 1) // 2  # all the edges possible
    num_collisions = int(P * num_possible_edges)  # pair of nodes to collide

    # pick up the collision pairs randomly
    all_edges = list(itertools.combinations(nodes, 2))
    sampled_edges = np.random.choice(len(all_edges), size=num_collisions, replace=False)
    return [all_edges[i] for i in sampled_edges]

def direct_sample_collisions(nodes, P):
    """
    direct sampling of collision pairs, avoiding explicit iteration over all pairs of nodes
    """
    N = len(nodes)
    num_possible_edges = N * (N - 1) // 2
    num_collisions = int(P * num_possible_edges)

    sampled_indices = np.random.choice(num_possible_edges, size=num_collisions, replace=False)

    edges = []
    for idx in sampled_indices:
        i = int((-1 + np.sqrt(1 + 8 * idx)) // 2)  # inverse of the formula for the i-th triangular number
        j = idx - i * (i + 1) // 2
        edges.append((nodes[i], nodes[j]))
    return edges

# A dynamic network model with time steps
def dynamic_network_model(G, iterations, P, plow, new_fragments_per_collision, P_remove_debris):
    avg_degree_log = []
    gc_size_log = []
    
    for t in range(iterations):
        nodes = list(G.nodes)
        if len(nodes) > 30000:
            return avg_degree_log, gc_size_log

        G = nx.empty_graph(len(nodes))
        collision_edges = direct_sample_collisions(nodes, P)
        G.add_edges_from(collision_edges)

        largest_cc = max(nx.connected_components(G), key=len)
        gc_size = len(largest_cc) / N  # Normalize GC size by N
        avg_degree = np.mean([d for _, d in G.degree()])

        avg_degree_log.append(avg_degree)
        gc_size_log.append(gc_size)

        for u, v in collision_edges:
            for _ in range(new_fragments_per_collision):
                if np.random.rand() < plow:
                    new_node = len(G.nodes)
                    G.add_node(new_node)

        if P_remove_debris > 0:
            max_degree = max([d for _, d in G.degree()], default=1)
            nodes_to_remove = [
                node for node, degree in G.degree() 
                if degree > 0 and np.random.rand() < (1 - np.exp(-P_remove_debris * (degree / max_degree)))
            ]
            G.remove_nodes_from(nodes_to_remove)

    return avg_degree_log, gc_size_log

# Set parameters
N = 1000  # Initial number of nodes
P = 0.0008  # Collision probability
plow = 0.01  # Probability of generating new fragments
new_fragments_per_collision = 2  # Debris per collision
iterations = 100  # Number of time steps
num_runs = 10  # Number of independent runs

# Different debris removal probabilities
probabilities = [0.005]

# Store all (K, S) points across runs
all_K = []
all_S = []

for _ in range(num_runs):
    for p in probabilities:
        G = nx.empty_graph(N)
        avg_degree_log, gc_size_log = dynamic_network_model(G, iterations, P, plow, new_fragments_per_collision, p)
        
        # Collect (K, S) points for each time step
        all_K.extend(avg_degree_log)
        all_S.extend(gc_size_log)

# Sort by K for a smooth curve
sorted_indices = np.argsort(all_K)
K_sorted = np.array(all_K)[sorted_indices]
S_sorted = np.array(all_S)[sorted_indices]

# Plot GC proportion vs. Average Degree
plt.figure(figsize=(10, 6))
plt.plot(K_sorted, S_sorted, marker='o', color='purple', linestyle='-', label='GC Proportion')
plt.title("Relationship Between Average Degree (K) and GC Proportion (S)")
plt.xlabel("Average Degree (K)")
plt.ylabel("GC Proportion (S)")
plt.grid(True)
plt.legend()
plt.show()
 

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import sys
import time

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
        write(0, 0, 0, 0, 0, N, 0, p, gc_size2, avg_degree2, "../../results", "static", "static")

def sample_collisions(nodes, P):
    """
    fast generation of collision pairs, avoiding explicit iteration over all pairs of nodes
    """
    N = len(nodes)
    num_possible_edges = N * (N - 1) // 2  # all the edges possible
    num_collisions = int(P * num_possible_edges)  # determine number of collisions to simulate

    # pick the collision pairs randomly
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

def satellite_launch(G, nr_sat_launches):
    """
    Adds new satellites (nodes) to the graph every launch_freq time steps.
    """
    N = len(G.nodes)
    new_nodes = range(N, N + nr_sat_launches)
    G.add_nodes_from(new_nodes)
    
    return G
    

# A dynamic network model with time steps
def dynamic_network_model(G, iterations, P, plow, new_fragments_per_collision, nr_sat_launches, launch_freq):
    """
    Simulates a dynamic network model with time steps, where nodes represent satellites or debris, 
    and edges represent potential collisions. The network evolves through collisions and satellite launches.

    Parameters:
    -----------
    G : networkx.Graph
        The initial network graph, where nodes represent satellites or debris.

    iterations : int
        The number of simulation steps to run.

    P : float
        The probability of a collision occurring between two nodes in the network.

    plow : float
        The probability that a collision generates a new fragment (debris).

    new_fragments_per_collision : int
        The number of new fragments generated per collision when plow is satisfied.

    nr_sat_launches : int
        The number of new satellites introduced into the system at each launch event.

    launch_freq : int
        The frequency (in time steps) at which new satellites are launched.

    Returns:
    --------
    avg_degrees : list of float
        A list of average node degrees over the simulation time steps.

    gc_proportions : list of float
        A list of proportions of the largest connected component (GC) relative to the total number of nodes.

    satellites_launched : list of int
        A list tracking the cumulative number of satellites launched at each time step.

    Notes:
    ------
    - If the number of nodes exceeds 3000, the simulation stops early to prevent excessive computational load.
    - Collisions are sampled based on probability `P`, and new fragments may be generated.
    - A record of the system's evolution is saved to the `results` directory using the `write` function.
    - New satellites are launched periodically according to `launch_freq`.

    Example Usage:
    --------------
    ```
    G = nx.erdos_renyi_graph(100, 0.1)  # Create an initial random graph
    avg_degrees, gc_proportions, satellites_launched = dynamic_network_model(G, 100, 0.05, 0.2, 2, 5, 10)
    ```
    """

    # current time
    current_time = time.time()

    # Initialize lists to track results
    avg_degrees = []
    gc_proportions = []
    cumulative_satellites = 0  # Track the cumulative number of satellites launched
    satellites_launched = []  # List to store the cumulative satellites launched at each timestep


    for t in range(iterations):
        # generate collision pairs
        nodes = list(G.nodes)
        
        if len(G.nodes) > 3000:
            print(f"For initial prob = {P}, the Number of nodes = {len(nodes)}, so stop simluations")
            return avg_degrees, gc_proportions, satellites_launched
        
        G = nx.empty_graph(len(nodes))

        collision_edges = direct_sample_collisions(nodes, P)
        G.add_edges_from(collision_edges)

        # GC size and average degree
        largest_cc = max(nx.connected_components(G), key=len)
        gc_size = len(largest_cc)
        avg_degree = np.mean([d for _, d in G.degree()])
        write(t+1, current_time, 0, 0, 0, len(G.nodes), 0, P, gc_size, avg_degree, "../../results", "dynamic", "dynamic")
        
       # Append results
        avg_degrees.append(avg_degree)
        gc_proportions.append(gc_size / len(G.nodes))  # Proportion of GC size relative to N

        # generate new fragments
        for u, v in collision_edges:
            for _ in range(new_fragments_per_collision):
                if np.random.rand() < plow:
                    new_node = len(G.nodes)
                    G.add_node(new_node)
                    # print(f"New fragment {new_node} generated from collision between nodes {u} and {v}, total nodes = {len(G.nodes)} \n")

        if t % launch_freq == 0:
            satellite_launch(G, nr_sat_launches)
            print(f"In time step {t}, {nr_sat_launches} new satellites were launched, total nodes = {len(G.nodes)} \n")
            cumulative_satellites += nr_sat_launches  # Update the cumulative count
        satellites_launched.append(cumulative_satellites)  # Track the count at each timestep    
    
    print("Returning results: avg_degrees, gc_proportions, and satellites_launched")
    

    return avg_degrees, gc_proportions, satellites_launched


def debris_removal_network(G, iterations, P, plow, new_fragments_per_collision, removal_rate):

    """
    Simulates a dynamic debris removal network where collisions generate new fragments, 
    and debris removal efforts reduce the number of debris at regular intervals.

    Parameters:
    -----------
    G : networkx.Graph
        The initial network graph, where nodes represent satellites or debris.

    iterations : int
        The number of simulation steps to run.

    P : float
        The probability of a collision occurring between two nodes in the network.

    plow : float
        The probability that a collision generates a new fragment (debris).

    new_fragments_per_collision : int
        The number of new fragments generated per collision when plow is satisfied.

    removal_rate : int
        The number of debris removed every 10 time steps. 
        The removal process stops after half of the total iterations.

    Returns:
    --------
    None
        The function modifies the graph in place and logs results to the `results` directory.

    Notes:
    ------
    - If the number of nodes exceeds 12,000, the simulation stops early to prevent excessive computational load.
    - Debris removal happens every 10 time steps and stops after half of the iterations.
    - New fragments are generated through collisions with a probability `plow`.
    - A record of the system’s evolution is saved to the `results` directory using the `write` function.

    Example Usage:
    --------------
    ```
    G = nx.erdos_renyi_graph(100, 0.1)  # Create an initial random graph
    debris_removal_network(G, 200, 0.05, 0.2, 3, 5)
    ```
    """

    # current time
    current_time = time.time()

    for t in range(iterations):
        # generate collision pairs
        nodes = list(G.nodes)
        nodes_number = len(nodes)
        if nodes_number > 12000:
            print(f"For initial prob = {P}, the Number of nodes = {len(nodes)}, so stop simluations")
            return
        
        # Just random clean the debris every 10 time steps
        # After half of the iterations, Clean robot power off.
        removal_number = 0
        if t < iterations /2 and t % 10 == 0 and nodes_number > 2 + removal_rate:
            nodes_number -= removal_rate
            removal_number = removal_rate

        # recover the graph but keep the massive debris as the new nodes
        G = nx.empty_graph(nodes_number)

        collision_edges = direct_sample_collisions(nodes, P)
        G.add_edges_from(collision_edges)

        # GC size and average degree
        largest_cc = max(nx.connected_components(G), key=len)
        gc_size = len(largest_cc)
        if gc_size == 1:
            gc_size = 0

        avg_degree = np.mean([d for _, d in G.degree()])
        write(t+1, current_time, 0, 0, 0, len(G.nodes), removal_rate, P, gc_size, avg_degree, "../../results", "debris", "removal")
        
        # generate new fragments
        for u, v in collision_edges:
            for _ in range(new_fragments_per_collision):
                if np.random.rand() < plow:
                    new_node = len(G.nodes)
                    G.add_node(new_node)
                    print(f"New fragment {new_node} generated from collision between {u} and {v}, total nodes = {len(G.nodes)} \n")

if __name__ == "__main__":

    # initial parameters
    N = 1000  # nodes
    P = 0.0008  # collision probability
    plow = 0.01 # probability of generating new fragments
    new_fragments_per_collision = 2  # debris per collision 
    iterations = 2000  # number of iterations, time steps

    # # genreate a probability list from 0.001 to 0.0009
    # probabilities = np.linspace(0.0001, 0.0009, 10)
    
    # # static data
    # static_network_model(N, probabilities)
    
    # # dynamic network over time
    # for p in probabilities:
    #     G = nx.empty_graph(N)
    #     dynamic_network_model(G, iterations, p, plow, new_fragments_per_collision)
    
    # debris removal network
    P0 = 0.0003
    rate = N**2 * P0 * plow
    alpha_list = np.linspace(0, 14, 7)
    for alpha in alpha_list:
        G = nx.empty_graph(N)
    debris_removal_network(G, iterations, P0, plow, new_fragments_per_collision, int(15) * int(rate))

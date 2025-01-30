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
        write(0, 0, 0, 0, 0, N, p, gc_size2, avg_degree2, "../../results", "static", "static")

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
        write(t+1, current_time, 0, 0, 0, len(G.nodes), P, gc_size, avg_degree, "../../results", "dynamic", "dynamic")
        
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


if __name__ == "__main__":

    # initial parameters
    N = 1000  # nodes
    P = 0.0008  # collision probability
    plow = 0.01 # probability of generating new fragments
    new_fragments_per_collision = 2  # debris per collision 
    iterations = 100  # number of iterations, time steps
    launch_freq = 5 # determines after how many timesteps a satellite is launched
    nr_sat_launches = 2 # number of satellites launched

    # # genreate a probability list from 0.001 to 0.0009
    # probabilities = np.linspace(0.0001, 0.0009, 10)
    
    # # static data
    # static_network_model(N, probabilities)
    
    # dynamic network over time
    # Initialize the network
    G = nx.empty_graph(N)
    avg_degrees, gc_proportions, satellites_launched = dynamic_network_model(
        G, iterations, P, plow, new_fragments_per_collision, nr_sat_launches, launch_freq)

    print(len(satellites_launched))
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(avg_degrees, gc_proportions, marker='o', color='purple', label='GC Proportion')
    plt.title("Relationship Between Average Degree (K) and GC Proportion (S)")
    plt.xlabel("Average Degree (K)")
    plt.ylabel("GC Proportion (S)")
    plt.grid(True)
    plt.legend()
    plt.show()
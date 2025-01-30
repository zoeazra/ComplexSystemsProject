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
def dynamic_network_model(G, iterations, P, plow, new_fragments_per_collision):
    # current time
    current_time = time.time()

    for t in range(iterations):
        # generate collision pairs
        nodes = list(G.nodes)
        if len(nodes) > 12000:
            print(f"For initial prob = {P}, the Number of nodes = {len(nodes)}, so stop simluations")
            return

        # recover the graph but keep the massive debris as the new nodes
        G = nx.empty_graph(len(nodes))

        collision_edges = direct_sample_collisions(nodes, P)
        G.add_edges_from(collision_edges)

        # GC size and average degree
        largest_cc = max(nx.connected_components(G), key=len)
        gc_size = len(largest_cc)
        avg_degree = np.mean([d for _, d in G.degree()])
        write(t+1, current_time, 0, 0, 0, len(G.nodes), 0, P, gc_size, avg_degree, "../../results", "dynamic", "dynamic")
        
        # generate new fragments
        for u, v in collision_edges:
            for _ in range(new_fragments_per_collision):
                if np.random.rand() < plow:
                    new_node = len(G.nodes)
                    G.add_node(new_node)
                    print(f"New fragment {new_node} generated from collision between {u} and {v}, total nodes = {len(G.nodes)} \n")


def debris_removal_network(G, iterations, P, plow, new_fragments_per_collision, removal_rate):
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
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import networkx as nx
import numpy as np

def sir(graph, initial_infected, params, max_steps):
    beta, gamma = params
    for node in graph.nodes():
        graph.nodes[node]['state'] = 'S'
    for node in initial_infected:
        graph.nodes[node]['state'] = 'I'

    infected_nodes = [initial_infected.copy()]
    
    for step in range(max_steps):
        new_state = {}
        
        for node in graph.nodes():
            if graph.nodes[node]['state'] == 'S':
                infected_neighbors = [n for n in graph.neighbors(node) if graph.nodes[n]['state'] == 'I']
                if infected_neighbors and np.random.rand() < 1 - (1 - beta) ** len(infected_neighbors): # probability that node gets infected by at least one of its infected neighbors
                    new_state[node] = 'I'
            elif graph.nodes[node]['state'] == 'I':
                if np.random.rand() < gamma:
                    new_state[node] = 'R'
        
        for node, state in new_state.items():
            graph.nodes[node]['state'] = state
        
    infected_nodes.append([node for node in graph.nodes() if graph.nodes[node]['state'] == 'I'])

    return infected_nodes

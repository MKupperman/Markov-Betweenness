import pandas as pd
import numpy as np
import igraph
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations
from numba import jit, njit, prange
import timeit

def rw_betweenness(graph, beta, verbose = False, num_samples=3):
    '''
    A top level function we can iterate over for all pairs of (source, target)
    Returns a list of incidence weights for all vertices that we hit on our many random walks, 
    divided by the total number of paths sampled. 
    
    '''
    
    assert(type(graph) is igraph.Graph), 'The graph object must be an igraph Graph object'
    assert(len(graph.clusters(mode='weak').sizes()) == 1), 'There is more than one compartment in the graph'

    #Let's precompute a few values we can use later
    vertex_set = graph.vs().indices
    graph_size = graph.vcount()
    vertex_pairs = list(combinations(vertex_set, 2))
    
    #Remove any possible duplicates
    for vertex_pair in vertex_pairs:
        if vertex_pair[0] == vertex_pair[1]:
            vertex_pairs.remove(vertex_pair)
            if verbose:
                print('We removed a vertex')
    
    #Setup a numpy array
    rw_betw = np.zeros(shape=(graph_size))
    
    for vertex_pair_index in vertex_pairs:
        results, num_samples = rw_sampler(graph=graph, graph_size=graph_size,
                                          start=vertex_pair[0], end=vertex_pair[1], beta=beta, 
                                          num_samples = num_samples, verbose=False, eventually_break=True,
                                          steps_max=10000, count_endpoints = False)
        # Iterate over the keys of results (a defaultdict) to increment the totals
        for key in results.keys():
            rw_betw[key] += results[key]/num_samples 
        
    return rw_betw


def rw_sampler(graph, graph_size, start, end,beta, num_samples = 100, verbose=False, eventually_break=True,
               steps_max=10000, count_endpoints = False):
    #Sample the space and return a weighted rw_betweenness vector not normalized.
    # Returns a dict where each key is the index and the value at the key is the count.
    results = defaultdict(int)
    
    for _ in range(num_samples):
        path = rw_path(graph=graph, start=start, end=end, beta=beta, verbose=verbose,
                       eventually_break=eventually_break, count_endpoints=count_endpoints)
        for vertex in path:
            results[vertex] += 1
        
    return results, num_samples


def rw_path(graph, start, end, beta, verbose=False, eventually_break=True, steps_max=10000, count_endpoints = False):
    '''
    #start: The initial vertex index
    #end: the vertex we want to find on our random walk
    
    Examples:
    rw_path(graph=network_sg, start=24, end=10, beta=beta, verbose=False)
    '''
    
    steps_taken = 0
    path = list() # Don't count the starting vertex in our centrality
    if count_endpoints:
        path.append(start)
    current_position = start
    while current_position != end:
        next_part = rw_step(graph=graph, source=current_position, target=end, beta=beta, verbose=verbose)
        path.append(next_part)
        steps_taken += 1
        if verbose:
            print('We moved from ' + str(current_position) + ' to ' + str(next_part))
        current_position = next_part
        
        if eventually_break and steps_taken >= steps_max:
            print('The random walk from ' + str(current_position) + ' to ' + str(next_part) + 'did not find the target.')
            break
    if not count_endpoints:
        path.remove(current_position)  # We don't want to count landing on the path as 
    return path


def rw_step(graph, source, target, beta, verbose = False):
    '''
    Determine the probability of a single step, returns new source index
    
    #beta: the probability of taking a step on a geodesic path, must be >=0, 
        * beta > 1 will act as if beta=1 and always take the geodesic path   
        
        Examples:
        rw_step(graph=network_sg, source=3, target=10, beta=beta, verbose=True)
        rw_step(graph=network_sg, source=14, target=10, beta=beta, verbose=True)
    '''
    
    choices = graph.neighbors(source)
    geo_paths = graph.get_all_shortest_paths(v=source, to=target)  # Default mode is out
    # Extract the unique first elements
    beta_options = list() # Should be the length of geo_paths
    
    #Optimize
    for path in geo_paths:
        if path[1] not in beta_options: # No duplicates
            beta_options.append(path[1]) #Sort these values into beta_options
            choices.remove(path[1]) #remove from alpha_options
    
    #Let's get the vertex indexes for the alpha level selection
    alpha_options = list()
    for node_index in choices:
        if node_index not in alpha_options:
            alpha_options.append(node_index)

    # if we have one option, we have to take it
    if len(alpha_options) == 0:  # We have exactly one [possible choice]       
        new_source = np.random.choice(beta_options, size=1)  # Our geodesic path always has a value
    elif np.random.sample() < beta:
        new_source = np.random.choice(beta_options, size=1)
        if verbose:
            print('Drew Beta option')
    else:
        new_source = np.random.choice(alpha_options, size=1)
        if verbose:
            print('Drew Alpha option')
    if verbose:
        print(beta_options, alpha_options)
    return new_source[0]

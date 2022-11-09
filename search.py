from random import random

import numpy as np
import itertools
import networkx as nx


def find_neighbors(tri, nodes, query):
    qidx = nodes.index(query)

    try:

        nslice = slice(
            tri.vertex_neighbor_vertices[0][qidx],
            tri.vertex_neighbor_vertices[0][qidx + 1],
        )

        idx = tri.vertex_neighbor_vertices[1][nslice]
    except:
        idx = list(tri[qidx])

    return [nodes[i] for i in idx]


def find_neighbors_n_hop(tri, nodes, query, n_hop=1):
    neighbors = find_neighbors(tri, nodes, query)
    search_nodes = list(neighbors)
    for hop in range(1, n_hop):
        new_nodes = []
        while search_nodes:
            new_query = search_nodes.pop(0)
            new_nodes.extend(find_neighbors(tri, nodes, new_query))
        search_nodes = new_nodes
        neighbors.extend(new_nodes)

    # remove duplicates (note, ``Node`` is unhashable)
    unique_neighbors = []
    [unique_neighbors.append(obj) for obj in neighbors if obj not in unique_neighbors]
    return unique_neighbors


def identify_motifs(nodes, graph, query, n_hop=1, random_sample=True):
    """Identify graph motifs given a subset of the nodes
    nodes : list of all relevant nodes
    graph : full graph
    query : sample subgraph from which to find the motif (list of nodes in the input query)
    n_hop : number of hops for neighbourhood of node
    """

    # subgraphs = get_neighbour_subgraph(graph,nodes,query,n_hop)
    sampled_graphs = sampling_query(query, 3)
    subgraphs = [graph.subgraph(item) for item in sampled_graphs]
    print("got subgraphs")

    # given the subgraphs identify a motif within the interest
    motif = retrieve_motif(subgraphs, query)
    print("got motif: ")
    print(motif.nodes)
    query_nodes = None

    if random_sample:
        extra_nodes = random.sample(range(100, len(graph.nodes)), 10) + random.sample(range(0, 100), 10)
        print("extra_nodes: ")
        print(extra_nodes)
        query_nodes = graph.subgraph(extra_nodes)

    # apply the subgraph matching to seed nodes from the full graph
    all_candidates = find_motif_in_graph(motif, graph, nodes, query_nodes, n_hop=1)

    return motif, all_candidates


def sampling_query(query, sample_size):
    sm = sample_size
    # for every subgraph of size sm of the query graph, check if isomorphic to motif
    # if yes, count +=1
    product_sm = []
    for i in range(0, sm):
        product_sm.append(query.nodes)
        # sampling the query graph into all the possible chunks of size sm
    sample = list(itertools.product(*product_sm))
    sample = [trip for trip in sample if len(list(set(trip))) == sm]

    u_sample = []
    for triplet in sample:
        a = list(triplet)
        a.sort()
        if not u_sample.count(a):
            u_sample.append(a)

    return u_sample


def find_motif_in_graph(motif, graph, nodes, query_nodes=None, n_hop=1):
    """ Search for copies of the motif on the graph
    Input
    motif : template to search along the graph
    graph : full graph
    n_hop : number of hops for neighbourhood of each node
    query_nodes : list of pre-selected nodes which could potentially fit the motif

    Output
    sample_list : list of all the subgraphs that are isomorphic to the motif
    """

    # check for features of nodes in motif
    # for each node in the graph, check if features match the motif_node features
    # if features match, check if motif
    sample_list = []
    size_motif = len(motif)

    if query_nodes is not None:
        subgraphs_temp = get_neighbour_subgraph(graph, nodes, query_nodes, n_hop)
        subgraphs = []
        i = 0
        for subgraph in subgraphs_temp:
            subgraph_unfrozen = nx.Graph(subgraph)
            subgraph_unfrozen.add_node(list(query_nodes.nodes)[i])
            i += 1
            subgraphs.append(subgraph_unfrozen)
    else:
        subgraphs = get_neighbour_subgraph(graph, nodes, graph, n_hop)

    for subgraph in subgraphs:
        sm = len(motif.nodes)
        # for every subgraph of size sm of the query graph, check if isomorphic to motif
        # if yes, count +=1
        product_sm = []
        for i in range(0, sm):
            product_sm.append(subgraph.nodes)
        # sampling the query graph into all the possible chunks of size sm
        sample = list(itertools.product(*product_sm))
        sample = [trip for trip in sample if len(list(set(trip))) == sm]

        u_sample = []
        for triplet in sample:
            a = list(triplet)
            a.sort()
            if not u_sample.count(a):
                u_sample.append(a)

        count = 0
        for trip in u_sample:
            trip.sort()
            sub_gr = graph.subgraph(trip)
            if edges_exist(graph, sub_gr):
                if nx.could_be_isomorphic(sub_gr, out_motif) and check_features(sub_gr, out_motif):
                    sample_list.append(sub_gr)

    return sample_list


def edges_exist(graph, subgraph):
    """ Checks whether the mentioned nodes are directly connected in the general graph."""
    count_edges = 0
    list_nodes = list(subgraph.nodes)
    for i in range(0, len(list_nodes) - 1):
        for j in range(i + 1, len(list_nodes)):
            if graph.has_edge(list_nodes[i], list_nodes[j]):
                count_edges += 1
    if count_edges >= (len(list_nodes) - 1):
        return True
    else:
        return False


def check_features(subgraph, motif):
    """place holder function for feature comparison. in the sample case, only checks if sum(features) is 5"""
    value = [sum(subgraph.nodes[i]['features']) for i in list(subgraph.nodes)]

    if sum(value) == len(subgraph) * 5:
        return True
    else:
        return False


def get_neighbour_subgraph(graph, nodes, query, n_hop=1):
    """ Retrief the subgraph of neighboring nodes for each node in the query.
    Input
    nodes : list of all relevant nodes
    graph : full graph
    query : sample subgraph
    n_hop : number of hops for neighbourhood of node

    Output
    subgraphs : subgraphs for the neighbouring nodes
    """
    subgraphs = []

    # iterate over each query node and find the n_hop neighbors
    for node in query.nodes:
        neighbors = find_neighbors_n_hop(graph, nodes, nodes[node], n_hop)
        # neighbors is a list of Node, but we need the list of indices for each node in the graph
        neighbors_index = [nodes.index(neighbor) for neighbor in neighbors]

        # store each one as a subgraph
        subgraph = graph.subgraph(neighbors_index)  # assuming graph is nx.Graph()
        subgraphs.append(subgraph)

    return subgraphs


def retrieve_motif(motif_list, query_graph):
    """ Identify the motif given a sample graph
    Input
    motif_list : list of candidates
    query_graph : list of graph from where the motifs are retrieved

    Output:
    out_motif : motif with the highest repeat within the query_graph from the motif_list
    """

    min_count = -1
    nodes = query_graph.nodes()
    out_motif = None
    for motif in motif_list:
        sm = len(motif.nodes)
        # for every subgraph of size sm of the query graph, check if isomorphic to motif
        # if yes, count +=1
        product_sm = []
        for i in range(0, sm):
            product_sm.append(nodes)
        # sampling the query graph into all the possible chunks of size sm
        sample = list(itertools.product(*product_sm))
        sample = [trip for trip in sample if len(list(set(trip))) == sm]
        sample = map(list, map(np.sort, sample))
        u_sample = []
        [u_sample.append(trip) for trip in sample if not u_sample.count(trip)]
        count = 0
        for trip in u_sample:
            sub_gr = query_graph.subgraph(trip)
            if edges_exist(query_graph, sub_gr):
                if nx.could_be_isomorphic(sub_gr, motif):
                    count += 1

            # count motif in the subgraph
        if count > min_count:
            min_count = count
            out_motif = motif

    return out_motif

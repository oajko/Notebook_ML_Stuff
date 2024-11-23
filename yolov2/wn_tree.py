from nltk.corpus import wordnet as wn
import networkx as nx
import matplotlib.pyplot as plt

root = wn.synset('physical_object.n.01')

def looper(graph = None, word = 'object', depth = 0, max_depth = 5):
    if graph is None:
        graph = nx.DiGraph()
    if depth > max_depth:
        x = {i: idx for idx, i in enumerate({node: list(graph.successors(node)) for node in graph.nodes()})}
        # Dictionary for indexing and predecessors (above funcs)
        return {i: idx for idx, i in enumerate({node: list(graph.successors(node)) for node in graph.nodes()})}, {node: list(graph.predecessors(node)) for node in graph.nodes()}
    for next in word.hyponyms():
        graph.add_edge(word.name().split('.')[0], next.name().split('.')[0])
        looper(graph, next, depth + 1, max_depth)
# looper()
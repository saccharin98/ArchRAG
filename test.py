import networkx as nx
import pandas as pd
from src.utils import read_graph_nx
g, ents, rels = read_graph_nx("index")
print("sample node attrs:", list(g.nodes(data=True))[:3])
print("entity columns:", ents.columns.tolist())
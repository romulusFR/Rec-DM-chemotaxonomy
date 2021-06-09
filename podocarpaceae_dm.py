"""









https://www.catalogueoflife.org/data/taxon/627HD

"""

# TODO : vérifier si j'arrive à retrouver les 3 familles
# monoterpenoids sesquiterpenoids diterpenoids ?

# %%

from collections import defaultdict
from pprint import pprint

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.cluster import SpectralCoclustering
from sklearn.cluster import SpectralClustering

# if some conversion from/to greek is needed
import transliterate


sns.set_theme(style="ticks")

# binarization threshold is 1/1000 of col based relative values
THRESHOLD = 0.1
# seed for randomized algs.
SEED = 0

# %%
df = pd.read_csv("data/podocarpaceae.csv", index_col="Compound")
df_col = 100 * df / df.sum(axis=0)
df_bin = df_col > THRESHOLD

df.info()

# Index: 95 entries, β-Cedrene to Rosa-5,15-diene
# Data columns (total 11 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   Ap      95 non-null     float64
#  1   Ft      95 non-null     float64
#  2   Da      95 non-null     float64
#  3   Db      95 non-null     float64
#  4   Dg      95 non-null     float64
#  5   Dl      95 non-null     float64
#  6   Rc      95 non-null     float64
#  7   Rm      95 non-null     float64
#  8   Ps      95 non-null     float64
#  9   Pn      95 non-null     float64
#  10  Pl      95 non-null     float64

species = df.columns
compounds = df.index

M = df.to_numpy()

# shape is 95x95
MC = M @ M.T

# shape is 11x11
MS = M.T @ M

# %%
# https://stackoverflow.com/questions/61112420/pandas-to-bipartite-graph

#  B is the bipartite graphs compund/species
B = nx.Graph()
B.add_nodes_from(compounds, type="compound")
B.add_nodes_from(species, type="species")
B.add_weighted_edges_from((e[0], e[1], w) for e, w in df.stack().to_dict().items() if w)


# %%

ex_species = ["Ap", "Ft", "Da", "Db", "Dg", "Dl"]
ex_compounds = ["Limonene", "α-Pinene", "β-Pinene", "α-Terpineol"]
S = B.subgraph(ex_species + ex_compounds)

# pos = nx.bipartite_layout(B, compounds)
pos = nx.spring_layout(S)
node_color = ["lightgreen" if d == "compound" else "blue" for n, d in S.nodes(data="type")]
width = width = [2 * w for u, v, w in S.edges(data="weight")]


nx.nx.draw_networkx(
    S,
    pos=pos,
    node_color=node_color,
    width=width,
    node_size=10,
    with_labels=True,
)


# %%
# Spectral CO-clustering

N_CLUSTERS = 3

A = df.to_numpy().astype(float)
co_clustering = SpectralCoclustering(n_clusters=N_CLUSTERS, random_state=SEED).fit(A)

comp_clst_dict = {c: co_clustering.row_labels_[i] for i, c in enumerate(compounds)}
spec_clst_dict = {s: co_clustering.column_labels_[i] for i, s in enumerate(species)}


comp_part = defaultdict(list)
for key, value in comp_clst_dict.items():
    comp_part[value].append(key)

# %%


c_clustering = SpectralClustering(n_clusters=N_CLUSTERS, affinity="precomputed", random_state=SEED).fit(MC)
c_clustering.labels_


#%%
#https://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:15384
# limonene

# import libchebipy
# chebi_entity = ChebiEntity(15903, download_dir="chebi")
# pylint: skip-file

# %%
from itertools import product

from pprint import pprint
import numpy as np
import pandas as pd
import prince
import matplotlib.pyplot as plt

BOOLS = [False, True]
SELECTORS = ["w/o", "w/"]  # ordered as bools

counted = pd.read_csv("undo.csv", index_col=[0, 1, 2], header=[0, 1, 2])
counted


# %%

# TODO => factoriser l'extraction des sommes marginales
margin_idx = ("*", "Σ")
mcols = counted.groupby(level=1).sum().drop_duplicates().reset_index(drop=True)
mcols.index = pd.MultiIndex.from_tuples([margin_idx])
print(mcols)

mrows = counted.groupby(level=1, axis=1).sum().iloc[:, 0]
mrows.name = margin_idx
mrows = pd.DataFrame(mrows)
print(mrows)

# %%

# TODO => factoriser le sanity check
mrows_sum = mrows.groupby(level=1, axis=0).sum()
mcols_sum = mcols.groupby(level=1, axis=1).sum()
N = mrows_sum[margin_idx][0]
assert (mrows_sum == N).all().bool()
assert (mcols_sum == N).T.all().bool()
print(N)

counted / N
# %%

X =  counted.index.get_level_values(1).unique()
X
Y =  counted.columns.get_level_values(1).unique()
Y



# %%

# creation d'un modèle "plausible", id est, dont le test X avec la donnée d'origine est bas
def generate_from_counts(nb_papers):
    """TODO !!!"""
    s, l, d, w = [False] * (len(X) + len(Y))
    for paper in range(nb_papers):
        # choisir son vecteur de bools
        
        # Si 
        #   pour chaque x in X, on a une Bin
        #   pour chaque y in Y, on a une Bin
        # on est en SPMI


        yield (s, l, d, w)


pprint(list(generate_from_counts(20)))

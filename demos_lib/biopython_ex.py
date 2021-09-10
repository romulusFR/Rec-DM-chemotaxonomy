"""Main module"""

#%%

import csv
import logging
from collections import defaultdict
# from pprint import pprint
# from typing import Counter
from urllib.parse import quote
# from html import escape
# import transliterate
# from unidecode import unidecode
import pandas as pd

from tools import replace_greek_alphabet

# https://biopython.org/docs/latest/api/Bio.KEGG.REST.html
# https://www.kegg.jp/kegg/rest/keggapi.html
from Bio.KEGG import REST, Compound
from Levenshtein import distance

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# %%

# def remove_greek(s):
#     return transliterate.translit(s, "el", reversed=True)


df = pd.read_csv("data/podocarpaceae.csv", index_col="Compound")
# %%

# path = pathway : voie de syntèse

# raw_search = "Caryophyllene oxide"

def find_kegg_match(name, db = "cpd"):
    to_search = replace_greek_alphabet(name)

    # http://rest.kegg.jp/find/compound/%22limonene%22
    # result is a list of lines "cpd:id\tname1; name2; ..."
    query = REST.kegg_find(db, quote(to_search), option=None)
    results = query.read()

    results_dict = defaultdict(dict)
    # après plit, la dernière ligne est une ligne vide
    for line in csv.reader(results.split("\n"), delimiter="\t"):
        if line:
            identifier = line[0] # .removeprefix(f"{db}:")
            names = line[1].split("; ")
            results_dict[identifier]["names"] = names
            results_dict[identifier]["levenshtein"] = min(distance(to_search.lower(), name.lower()) for name in names)


    results_ordered = sorted(results_dict.items(), key=lambda x: x[1]["levenshtein"])  # type: ignore

    return results_ordered

#%%


# https://www.genome.jp/entry/ko00903

# https://www.genome.jp/entry/C06078
# limonene = find_kegg_match("limonene")
# alpha_pinene = find_kegg_match("α-Pinene")
# manool_oxide = find_kegg_match("Manool oxide")
# rho_cymene = find_kegg_match("ρ-Cymene")


ex_compounds = ["Limonene", "α-Pinene", "β-Pinene", "α-Terpineol", "Manool oxide", "ρ-Cymene"]

# https://github.com/biopython/biopython/blob/master/Bio/KEGG/Compound/__init__.py
# https://github.com/biopython/biopython/tree/master/Tests/KEGG
res = []
for comp_name in ex_compounds:
    r = find_kegg_match(comp_name, "cpd")
    if r:
        best_match = r[0]
        # attention au split ! c'est ligne par ligne !
        details = REST.kegg_get(best_match[0]).read().split("\n")
        compound = list(Compound.parse(details))[0]
    else:
        best = None
        compound = None

    res.append(compound)


# %%
# pour le S limonene
REST.kegg_get("C00521").read().split("\n")
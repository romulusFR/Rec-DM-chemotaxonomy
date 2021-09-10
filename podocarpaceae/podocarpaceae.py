"""Jupyter-notebook for chemotaxonomy of Podocarpaceae"""

# %%
# for pylint, unused in jupyter

# pylint: disable=pointless-string-statement
# pylint: disable=wrong-import-position
# pylint: disable=line-too-long
# pylint: disable=unused-import

# %% [markdown]
"""
Computer-aidded chemotaxonomy of Podocarpaceae
==============================================

This is exploratory data analysis for the [chemotaxonomy](https://en.wikipedia.org/wiki/Chemotaxonomy) of [podocarpaceae](https://en.wikipedia.org/wiki/Podocarpaceae) via their [terpenes](https://en.wikipedia.org/wiki/Terpene). See paper [Podocarpaceae][] and Podocarpacae (taxon 627HD) in the [catalogue of life](https://www.catalogueoflife.org/data/taxon/627HD)

[Podocarpaceae]: https://onlinelibrary.wiley.com/doi/pdf/10.1002/cbdv.201400445 "Chemical Diversity of Podocarpaceae in New Caledonia: Essential Oils from Leaves of Dacrydium, Falcatifolium, and Acmopyle Species, Nicolas Lebouvier, Leïla Lesaffre, Edouard Hnawia, Christine Goué, Chantal Menut, Mohammed Nour"

We use the [Bioservice][] Python module to access the following databases via their _web services_  :

- ChEBI <https://www.ebi.ac.uk/chebi/webServices.do>
- KEGG <https://www.kegg.jp/kegg/rest/keggapi.html>

[Bioservice]: https://bioservices.readthedocs.io/

The following compound are used as base examples:

- monoterpene : (S)-Limonene / (R)-Limonene
"""


#%%
# first of all, the libraries we need
import logging
from operator import itemgetter
from pprint import pprint


import pandas as pd

# remote access to databases
from bioservices.chebi import ChEBI
from bioservices.kegg import KEGG

# chemistry lib
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw

IPythonConsole.ipython_useSVG = True  # < set this to False if you want PNGs instead of SVG
# IPythonConsole.drawOptions.addStereoAnnotation = True

from Levenshtein import distance

from tools import parse_kegg_list, replace_greek_alphabet

# some basic constant/objects we'll need
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

chebi = ChEBI(verbose=True)
kegg = KEGG(verbose=True)


# %% [markdown] %%%%%%%%%%%%%% ChEBI %%%%%%%%%%%%%%

"""
For the first part, we use the ChEBI endpoint via bioservice's service, see [the doc](https://bioservices.readthedocs.io/en/master/references.html#bioservices.chebi.ChEBI)
"""

# %%
# full text search of "limonene"
search_limonene = chebi.getLiteEntity("limonene", searchCategory="CHEBI NAME", stars="THREE ONLY")

# full text search of "alpha terpineol"
search_alpha_terpineol = chebi.getLiteEntity("alpha terpineol", searchCategory="CHEBI NAME", stars="THREE ONLY")

# results are ordered by decreasing searchScore
# the first one is "CHEBI:15384" which stand for limonene (base compound, not enantiomere)
best_limonene_chebi_id = search_limonene[0].chebiId

# %%
# the whole ChEBI Entity

ALPHA_TERPINEOL_CHEBI_ID = "CHEBI:22469"
LIMONENE_CHEBI_ID = "CHEBI:15384"
LIMONENE_KEGG_ID = "C06078"
S_LIMONENE_KEGG_ID = "C00521"


limonene_chebi = chebi.getCompleteEntity(best_limonene_chebi_id)
assert limonene_chebi.chebiAsciiName == "limonene"
assert limonene_chebi.chebiId == LIMONENE_CHEBI_ID

# links to other databases
limonene_links_kegg = [x["data"] for x in limonene_chebi.DatabaseLinks if x["type"] == "KEGG COMPOUND accession"]
assert limonene_links_kegg[0] == LIMONENE_KEGG_ID

print(limonene_chebi.definition)

alpha_terpineol_chebi = chebi.getCompleteEntity(ALPHA_TERPINEOL_CHEBI_ID)
print(alpha_terpineol_chebi.definition)

# %%
# let's navigate a bit in ChEBI's ontology to find limonene's links
limonene_is_a_parents = [x for x in limonene_chebi.OntologyParents if x["type"] == "is a"]
pprint(limonene_is_a_parents)

limonene_is_a_children = [x for x in limonene_chebi.OntologyChildren if x["type"] == "is a"]
pprint(limonene_is_a_children)

# %%
# search by similar structure : OSEF ?
limonene_similar = chebi.getStructureSearch(
    limonene_chebi.smiles, mode="SMILES", structureSearchCategory="SIMILARITY", totalResults=50, tanimotoCutoff=0.90
)
assert len(limonene_similar.ListElement) == 48


# %%
# a top-down approach : find all terpenes
TERPENE_CHEBI_ID = "CHEBI:35186"
terpene_chebi = chebi.getCompleteEntity(TERPENE_CHEBI_ID)
terpene_all_children = chebi.getAllOntologyChildrenInPath(TERPENE_CHEBI_ID, "is a", True)

print(len(terpene_all_children.ListElement))  # 361 on 2021-05-19


# %% [markdown] %%%%%%%%%%%%%% KEGG %%%%%%%%%%%%%%

"""
Now, examples with the KEGG endpoint
"""

# %%
limonene_kegg_results_str = kegg.find("compound", "limonene")
# the length in characters
print(len(limonene_kegg_results_str))
# i wrote a small parser
limonene_kegg_results = parse_kegg_list(limonene_kegg_results_str)

pprint(limonene_kegg_results)

# %%
# KEGG's results are not ordered, so order them by Levenhstein distance

limonene_kegg_results_scores = {
    key: min(distance("limonene", name.lower()) for name in names) for key, names in limonene_kegg_results.items()
}

limonene_kegg_results_scores_sorted = [i[0] for i in sorted(limonene_kegg_results_scores.items(), key=itemgetter(1))]

assert limonene_kegg_results_scores_sorted[0] == "cpd:" + LIMONENE_KEGG_ID


# %%
# the whole KEGG entity as a python Dict
limonene_kegg = kegg.parse(kegg.get("cpd:" + LIMONENE_KEGG_ID))
# its S enantiomere
s_limonene_kegg = kegg.parse(kegg.get("cpd:" + S_LIMONENE_KEGG_ID))


print(s_limonene_kegg["PATHWAY"])

# %%
# biosynthesis pathways
terpenoid_pathways = kegg.find("pathway", "terpenoid")
print(terpenoid_pathways)

# Biosynthesis of terpenoids and steroids - Reference pathway
# https://www.kegg.jp/pathway/map01062


# %% [markdown] %%%%%%%%%%%%%% RDKIT %%%%%%%%%%%%%%

"""
Now, let's play with RDKit to draw the stuff
"""

# %%
print(limonene_chebi.smiles)
limonene_rdkit = Chem.MolFromSmiles(limonene_chebi.smiles)
alpha_terpineol_rdkit = Chem.MolFromSmiles(alpha_terpineol_chebi.smiles)

Draw.MolsToGridImage([limonene_rdkit, alpha_terpineol_rdkit])


# %% [markdown]

""".getCompleteEntityByList
Now, we use the chemical data
"""

# %%
# the base table
df = pd.read_csv("data/podocarpaceae.csv", index_col="Compound")
df.info()

species = df.columns
compounds = df.index


SOME_COMPOUNDS = ["Limonene", "α-Pinene", "β-Pinene", "α-Terpineol", "Manool oxide", "ρ-Cymene"]


# %%


def chebi_search_helper(name, searchCategory="CHEBI NAME", maximumResults=5, stars="THREE ONLY"):
    """Little helper"""
    return chebi.getLiteEntity(
        replace_greek_alphabet(name), searchCategory=searchCategory, maximumResults=maximumResults, stars=stars
    )


# WARNING : takes about 1min
# TODO : parallel !!!
chebi_bulk_search = {name: chebi_search_helper(name, stars="ALL") for name in compounds}


#%%
# enrich with levenshtein distance

for name in compounds:
    for entity in chebi_bulk_search[name]:
        d = float(distance(replace_greek_alphabet(name).lower(), entity.chebiAsciiName.lower()))
        logger.debug(f"comparing {name} and {entity.chebiAsciiName}: {d}")
        entity["levenshtein"] = d


# flatten the first dimension so each result is a row with columns
# 'search', 'chebiId', 'chebiAsciiName', 'searchScore', 'entityStar', 'levenshtein'
flat_chebi_bulk_search = [
    {"search": name} | dict(entity) for name, entities in chebi_bulk_search.items() for entity in entities
]

default_search = {
    "chebiId": "NOT FOUND",
    "chebiAsciiName": "NOT FOUND",
    "searchScore": 0,
    "entityStar": 0,
    "levenshtein": 00,
}
not_found_in_chebi = [{"search": name} | default_search for name, entities in chebi_bulk_search.items() if not entities]

chebi_searches = pd.DataFrame(flat_chebi_bulk_search + not_found_in_chebi)
chebi_searches.set_index(["search", "chebiAsciiName"])
chebi_searches.index.name = "id"
chebi_searches.sort_index()
chebi_searches.to_csv("data/podocarpaceae_chebi.csv")



# %%

# %%
from pybliometrics.scopus import ScopusSearch, CitationOverview, AbstractRetrieval
import pandas as pd


# %%


compound = "pyrrolizidine"  # alkaloid
activity = "antifungal"  # pharmaco
key = "KEY"
# 16 downloaded on 2022-01-29
# submatrix
#                                 pharmaco    pharmaco
#                                 antifungal  antifungal
#                                 w/o         w/
# alkaloid    pyrrolizidine   w/o 228463      14806
# alkaloid    pyrrolizidine   w/  679         16


query = f'DOCTYPE("ar") AND {key}({compound}) AND {key}({activity})'


# download=False => results size only
s = ScopusSearch(query=query, subscriber=False, verbose=True, download=True)
#


print(s.get_results_size())
# onli eids
print(s)
# named tuples
# print(s.results)
# 35 columns, when eid indexed
df = pd.DataFrame(s.results).sort_values("citedby_count", ascending=False)
print(df.columns)



# %%
most_cited = list(df["doi"].iloc[:3])
print(most_cited)

# %%
# 403
# co = CitationOverview(most_cited, start=2019, end=2021)

ab = AbstractRetrieval(most_cited[0], subscriber=False, verbose=True, view="META", id_type="doi")

print(ab.title)
print(ab.authors)
print(ab.idxterms)
print(ab.subject_areas)
print(ab.authkeywords)


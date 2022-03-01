# %%


import pandas as pd
from itertools import product, islice

BOOLS = [False, True]
SELECTORS = ["w/o", "w/"]  # ordered as bools

# read raw data file written as 3 + 4 = 7 bits
df = pd.read_csv("data/2_mrcv.txt", sep=" ")

# constants : rows and columns
wastes = [
    "Nitrogen",
    "Phosphorus",
    "Salt",
]
storages = ["Lagoon", "Pit", "Drainage", "Tank"]
df.columns = wastes + storages

# grand total
N = len(df)

assert df["Tank"].sum() == 13

# %%
# df for results
res = pd.DataFrame(
    index=pd.MultiIndex.from_tuples(list(product(wastes, SELECTORS))),
    columns=pd.MultiIndex.from_tuples(list(product(storages, SELECTORS))),
).fillna(0)

res


# res.loc["Salt", "Pit"]
# res.loc[w, s] = [[0,1],[2,3]]
# res.loc[("Salt", "w/"), ("Pit", "w/o")]

# %%
# for each combination (waste, storage), add 1 to corresponding celle of the subtable
res.loc[:] = 0
for c in df.itertuples():
# for c in islice(df.itertuples(),10):

    ws = [wastes[i] for i, v in enumerate(c[1 : len(wastes) + 1]) if v]
    ss = [storages[i] for i, v in enumerate(c[len(wastes) + 1 :]) if v]
    # print(c)
    # print(ws, ss)
    for w_i, w_p in enumerate(c[1 : len(wastes) + 1]):
        for s_j, s_p in enumerate(c[len(wastes) + 1 :]):
            # print([(wastes[w_i],SELECTORS[w_p]), (storages[s_j],SELECTORS[s_p])])
            res.loc[(wastes[w_i], SELECTORS[w_p]), (storages[s_j], SELECTORS[s_p])] +=1

# write in good format
# res.to_csv("data/2_mrcv.csv")

# %%
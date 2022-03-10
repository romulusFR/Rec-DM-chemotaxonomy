"""Using Bilder's MRCV waste dataset"""
# %%


from itertools import product

import numpy as np
import pandas as pd
import prince
import matplotlib.pyplot as plt


BOOLS = [False, True]
SELECTORS = ["w/o", "w/"]  # ordered as bools

# read raw data file written as 3 + 4 = 7 bits
raw = pd.read_csv("data/2_mrcv.txt", sep=" ")

# constants : rows and columns
wastes = [
    "Nitrogen",
    "Phosphorus",
    "Salt",
]
storages = ["Lagoon", "Pit", "Drainage", "Tank"]
raw.columns = wastes + storages

I, J = len(wastes), len(storages)
print(I, J)

# grand total
N = len(raw)

assert raw["Tank"].sum() == 13

# %%
# df for results
df = pd.DataFrame(
    index=pd.MultiIndex.from_tuples(list(product(wastes, reversed(SELECTORS)))),
    columns=pd.MultiIndex.from_tuples(list(product(storages, reversed(SELECTORS)))),
).fillna(0)


# res.loc["Salt", "Pit"]
# res.loc[w, s] = [[0,1],[2,3]]
# res.loc[("Salt", "w/"), ("Pit", "w/o")]

# %%
# for each combination (waste, storage), add 1 to corresponding celle of the subtable
df.loc[:] = 0
for c in raw.itertuples():
    # for c in islice(df.itertuples(),10):

    ws = [wastes[i] for i, v in enumerate(c[1 : len(wastes) + 1]) if v]
    ss = [storages[i] for i, v in enumerate(c[len(wastes) + 1 :]) if v]
    # print(c)
    # print(ws, ss)
    for w_i, w_p in enumerate(c[1 : len(wastes) + 1]):
        for s_j, s_p in enumerate(c[len(wastes) + 1 :]):
            # print([(wastes[w_i],SELECTORS[w_p]), (storages[s_j],SELECTORS[s_p])])
            df.loc[(wastes[w_i], SELECTORS[w_p]), (storages[s_j], SELECTORS[s_p])] += 1

print(df)

# %%

# write too csv with good format
# w/ w/o inverted !
# res.to_csv("data/2_mrcv.csv")

# %%

# marignal sums
margin_idx = ("*", "Σ")
margin_cols = df.groupby(level=0).sum().drop_duplicates().reset_index(drop=True)
margin_cols.index = pd.MultiIndex.from_tuples([margin_idx])
print(margin_cols)


margin_rows = df.groupby(level=0, axis=1).sum().iloc[:, 0]
margin_rows.name = margin_idx
margin_rows = pd.DataFrame(margin_rows)
print(margin_rows)

# margin_rows.loc[('Salt', 'w/o'),margin_idx]

# %%
# compute expectation from marginal sums :
# Simultaneous Pairwise Marginal Independence (SPMI)

ex = pd.DataFrame().reindex_like(df)


for r_idx in df.index:
    for c_idx in df.columns:
        r_margin = margin_rows.loc[r_idx, margin_idx]
        c_margin = margin_cols.loc[margin_idx, c_idx]
        # print(r_idx, r_margin, c_idx, c_margin)
        ex.loc[r_idx, c_idx] = r_margin * c_margin / N


# for w_i, w_p in enumerate(c[1 : len(wastes) + 1]):
#     for s_j, s_p in enumerate(c[len(wastes) + 1 :]):
#         # print([(wastes[w_i],SELECTORS[w_p]), (storages[s_j],SELECTORS[s_p])])
#         # df.loc[(wastes[w_i], SELECTORS[w_p]), (storages[s_j], SELECTORS[s_p])] += 1
#         ex.loc[(wastes[w_i], SELECTORS[w_p]), (storages[s_j], SELECTORS[s_p])] =


# %%


def odds_ratio(arr):
    """The trivial one. Just keep the bottom right cell. Amount to compute on "the old matrix" """
    [FF, FT], [TF, TT] = arr.reshape(2, 2)
    return TT * FF / TF * FT


# redimension the values to a 4D array
M_2_2 = np.moveaxis(df.values.reshape((I, 2, J, 2)), 1, -2)
M_4 = M_2_2.reshape((I * J, 4))

print(f"{M_2_2.shape = }")
print(f"{M_4.shape = }")


total = np.sum(M_2_2, axis=(2, 3), keepdims=False)
print(f"{np.all(total == N) = } (with {N = })")

# %%
# le cas "Nitrogen"/"Lagoon"

case_observed = df.loc["Nitrogen", "Lagoon"]
case_expected = ex.loc["Nitrogen", "Lagoon"]


print("Nitrogen/Lagoon observed:", odds_ratio(case_observed.values))
print("Nitrogen/Lagoon expected:", odds_ratio(case_expected.values))


# %%
values = np.moveaxis(df.values.reshape((I, 2, J, 2)), 1, -2).reshape((I * J, 4))
matrix = np.apply_along_axis(odds_ratio, 1, values).reshape((I, J))
observed_odds = pd.DataFrame(matrix, index=wastes, columns=storages)

values = np.moveaxis(ex.values.reshape((I, 2, J, 2)), 1, -2).reshape((I * J, 4))
matrix = np.apply_along_axis(odds_ratio, 1, values).reshape((I, J))
expected_odds = pd.DataFrame(matrix, index=wastes, columns=storages)

# %%
(100 * observed_odds / expected_odds).round()

# %%

chi_square = (observed_odds - expected_odds) ** 2 / expected_odds
# %%

ca = prince.CA(
    n_components=2,
    copy=True,
    check_input=True,
    engine="auto",
    random_state=42,
)
ca = ca.fit(chi_square)

# %%

ax = ca.plot_coordinates(
    X=chi_square,
    ax=None,
    figsize=(6, 6),
    x_component=0,
    y_component=1,
    show_row_labels=True,
    show_col_labels=True
)
plt.show()


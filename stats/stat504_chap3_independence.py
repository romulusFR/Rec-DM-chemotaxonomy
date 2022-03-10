#%%
# pylint: skip-file
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import power_divergence

FILENAME = Path("vitamin_c.csv")
df= pd.read_csv(FILENAME, index_col=0)
# .sort_index(axis=0).sort_index(axis=1)

df

# %%
margin_rows = df.sum(axis=1).values.reshape(2, 1)
margin_cols = df.sum(axis=0).values.reshape(1, 2)
N = df.sum().sum()

# div by rows
pb = df.values / N
pb


# %%
# https://online.stat.psu.edu/stat504/lesson/3/3.3
# produit des marges
observed = df.values
expected = margin_rows @ margin_cols / N
expected
#%%
chi = power_divergence(observed.reshape(4,), expected.reshape(4,), ddof=2, lambda_="pearson")
chi
#%%
div = power_divergence(observed.reshape(4,), expected.reshape(4,), ddof=2, lambda_="log-likelihood")
div


# %%
# https://online.stat.psu.edu/stat504/lesson/3/3.4

# difference in proportions, "row implication"
df.values / margin_rows

odds = (observed[0][0] * observed[1][1]) / (observed[1][0] * observed[0][1])
odds = 1 / odds

log_conf = 1.96*np.sqrt((1/observed).sum())
log_int = np.array([np.log(odds) + log_conf, np.log(odds) - log_conf])
print(odds, np.exp(log_int))
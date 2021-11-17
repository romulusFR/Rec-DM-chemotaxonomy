"""Some statistics"""

# pylint: disable=unused-import
# %%

from pathlib import Path
from itertools import islice
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chisquare, chi2_contingency, chi2

from loader import load_chemo_activities, sort_by_class

np.set_printoptions(precision=4, suppress=True)
# pd.set_option("display.float_format", lambda x: "{:.3f}".format(x))


FILENAME = Path("results/activities_2021-11-10_12-34-35.csv")
dataset = load_chemo_activities(FILENAME)

# converts to pandas
df = pd.DataFrame(dataset.data).T
df.columns.rename("Activity", inplace=True)
df.index.rename("Compound", inplace=True)

# %%

X = df.values
I = X.shape[0]
J = X.shape[1]

N = np.sum(X)
Z = X / N
print(f"Total number of citations {N = } with {I = } and {J = }")


R = np.sum(X, axis=1)  # X @ np.ones(X.shape[1])
print(f"{R = }")
C = np.sum(X, axis=0)  # Z @ np.ones(X.shape[0])
print(f"{C = }")


r = np.sum(Z, axis=1)
print(f"Marginal sums of rows ({np.sum(r) = }\n{100*r = }")
c = np.sum(Z, axis=0)
print(f"Marginal sums of columns ({np.sum(c) = })\n{100*c = }")

Dc = np.diag(c)
Dr = np.diag(r)

# theoretical distribution under the null hypothesis (independance of rows/columns)
T = r[:, np.newaxis] @ (c[:, np.newaxis].T)
print(f"{T.shape = }")
Zc = Z - T
S = np.diag(r ** (-0.5)) @ Zc @ np.diag(c ** (-0.5))

# %%


chi_square = ((X - N * T) ** 2) / (N * T)
print(f"{np.sum(chi_square) = }")

# wrapper for chi2_contingency
# chi2, p = chisquare(X.ravel(), f_exp=(N * T).ravel(), ddof=(I - 1) * (J - 1))

chi2_val, p, dof, ex = chi2_contingency(X, correction=False)
print(f"{chi2_val, p, dof = }")

assert dof == (I - 1) * (J - 1)
assert np.isclose(ex, N * T).all()


# %%
dof = (I - 1) * (J - 1)
rv_chemo = chi2(dof)

print(f"{chi2_val = }")
print(f"{rv_chemo.cdf(chi2_val) = }")

for e in range(1, 10):
    print(f"{1- 10 ** -e = } X² = {rv_chemo.ppf(1- 10 ** -e)}")

# # fig, ax = plt.subplots(1, 1)
# mean, var, skew, kurt = chi2.stats(df, moments="mvsk")
# x = np.linspace(chi2.ppf(0.001, df), chi2.ppf(0.999, df), 100)

# plt.plot(x, chi2.cdf(x, df), "r-", lw=5, alpha=0.6, label="chi2 pdf")

# %%

# from itertools import combinations
from math import factorial, prod
import numpy as np
from scipy.stats import multinomial


# https://scipy.github.io/devdocs/reference/generated/scipy.stats.multinomial.html?highlight=multinomial#scipy.stats.multinomial

N = 12
p = np.array([20, 15, 65]) / 100
rv = multinomial(N, p)
rv.mean()


def comb(n: int, p: int) -> int:
    """Number of combination of p among n"""
    if n < p:
        raise ValueError(f"{n = } < { p = }")
    if n < 0 or p < 0:
        raise ValueError(f"{n = } or { p = }")
    return factorial(n) // (factorial(p) * factorial(n - p))

    # return sum( 1 for _ in combinations())


# %%
# https://online.stat.psu.edu/stat504/lesson/2/2.3/2.3.1
# Let's find the probability that the jury contains:

#     three Black, two Hispanic, and seven Other members;
#     four Black and eight Other members;
#     at most one Black member.

ex1 = [3, 2, 7]
ex2 = [4, 0, 8]


pmf1_ref = rv.pmf(ex1)
print(pmf1_ref)

# à la main
pmf1_ref1 = (
    comb(N, ex1[0])
    * comb(N - ex1[0], ex1[1])
    * comb(N - ex1[0] - ex1[1], ex1[2])
    * (p[0] ** ex1[0])
    * (p[1] ** ex1[1])
    * (p[2] ** ex1[2])
)

assert np.isclose(pmf1_ref, pmf1_ref1)


def comb_vec(vs: list[int]) -> float:
    """le nombre de facon ce choisir le vecteur (de taille k) parmi k
    catégories

    comb_vec([p, n-p]) == comb(n, p)
    comb_vec(v) == comb_vec(v') poru toute permutation v' de v
    """
    return factorial(sum(vs)) / prod(factorial(v) for v in vs)


pmf1_ref2 = comb_vec(ex1) * prod(pi**e for pi, e in zip(p, ex1))
assert np.isclose(pmf1_ref, pmf1_ref2)

# %%
print(rv.pmf(ex2))

# /!\ en fait une binomiale : B = 1 , H | O = 0
# ex3 = [0, None, None] or [1, None, None]
p2 = np.array([p[0], sum(p[1:])])
bv = multinomial(N, p2)
print(bv.pmf([1, N - 1]) + bv.pmf([0, N - 0]))

# %%

cov = rv.cov()


cov_ref = N * np.array(
    [
        [p[0] * (1 - p[0]), -p[0] * p[1], -p[0] * p[2]],
        [-p[1] * p[0], p[1] * (1 - p[1]), -p[1] * p[2]],
        [-p[2] * p[0], -p[2] * p[1], p[2] * (1 - p[2])],
    ]
)
assert np.all(np.isclose(cov, cov_ref))
# %%

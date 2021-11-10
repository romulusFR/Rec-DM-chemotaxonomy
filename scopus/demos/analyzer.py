"""Analysis module"""

# pylint: disable=unused-import
# %%
import logging
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import numpy as np
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from downloader import BASE_CLASS, get_classes, COMPOUNDS, PHARMACOLOGY, sorted_keys, load_results

logging.basicConfig()
logger = logging.getLogger("scopus_api")
logger.setLevel(logging.DEBUG)

INPUT_MATRIX = Path("results/activity.csv")
CSV_PARAMS = {"delimiter": ";", "quotechar": '"'}

CHEMO_MAP = get_classes(COMPOUNDS)
PHARM_MAP = get_classes(PHARMACOLOGY)
CHEMO_CLS_NB = sum(1 for cls in CHEMO_MAP.values() if cls == BASE_CLASS)
PHARM_CLS_NB = sum(1 for cls in PHARM_MAP.values() if cls == BASE_CLASS)


chemo_cls = sorted_keys(CHEMO_MAP, base_only=True)
pharm_cls = sorted_keys(PHARM_MAP, base_only=True)
chemo_sub = sorted_keys(CHEMO_MAP, base_only=False)[CHEMO_CLS_NB::]
pharm_sub = sorted_keys(PHARM_MAP, base_only=False)[PHARM_CLS_NB::]


citations = load_results(
    INPUT_MATRIX, chemo_cls_nb=len(chemo_cls), pharm_cls_nb=len(pharm_cls), pharm_nb=len(PHARM_MAP)
)
[[chemo_cls_pharm_cls, chemo_cls_pharm_sub], [chemo_sub_pharm_cls, chemo_sub_pharm_sub]] = citations


# %%
# avec les classes

df_cls_cls = pd.DataFrame.from_records(chemo_cls_pharm_cls, index=chemo_cls, columns=pharm_cls)

pdf_cls_cls = pd.melt(
    df_cls_cls.reset_index(), ignore_index=True, var_name="pharmacology", value_name="nb", id_vars="index"
)
pdf_cls_cls.rename(columns={"index": "compound"}, inplace=True)


f, ax = plt.subplots(figsize=(12, 6))
sns.despine(f)
sns.histplot(pdf_cls_cls, discrete=True, multiple="stack", x="compound", hue="pharmacology", stat="count", weights="nb")


# %%
# avec les sujets
df_sub_sub = pd.DataFrame.from_records(chemo_sub_pharm_sub, index=chemo_sub, columns=pharm_sub)

# on filtre les lignes et les colonnes dont la somme vaut > 100
df_sub_sub = df_sub_sub.loc[df_sub_sub.sum(axis=1) > 100, df_sub_sub.sum(axis=0) > 100]


# on pivote
pdf_sub_sub = pd.melt(
    df_sub_sub.reset_index(), ignore_index=True, var_name="pharmacology", value_name="nb", id_vars="index"
)
pdf_sub_sub.rename(columns={"index": "compound"}, inplace=True)


# l'histogramme absolu
f, ax = plt.subplots(figsize=(12, 6))
sns.despine(f)



sns.histplot(
    pdf_sub_sub,
    discrete=True,
    multiple="stack",
    x="compound",
    hue="pharmacology",
    stat="count",
    weights="nb",
)

# %%
# matrice des correlation, Pearson par défaut

# pharm X pharm
df_sub_sub.corr()

# chemo X chemo
df_sub_sub.transpose().corr()


f, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(df_sub_sub.corr())


# %%

# l'histogramme relatif
sns.set_theme(style="ticks")
# ici on peut normaliser, ici par ligne
df_sub_sub_n = df_sub_sub.apply(lambda x: 100 * x / df_sub_sub.sum(axis=1))
pdf_sub_sub_n = pd.melt(
    df_sub_sub_n.reset_index(), ignore_index=True, var_name="pharmacology", value_name="nb", id_vars="index"
)
pdf_sub_sub_n.rename(columns={"index": "compound"}, inplace=True)

# palette = sns.color_palette("husl", len(pharm_sub))


f, ax = plt.subplots(figsize=(12, 6))
sns.despine(f)
plt.xticks(rotation=90)
sns.histplot(
    pdf_sub_sub_n,
    discrete=True,
    multiple="stack",
    x="compound",
    hue="pharmacology",
    stat="count",
    weights="nb",
)


# %%
f, ax = plt.subplots(figsize=(10, 8))
corr = df_sub_sub_n.corr()
mask = np.triu(np.ones_like(corr))

ax = sns.heatmap(corr, mask=mask)

# %%

from sklearn.cluster import SpectralClustering

NB_CLUSTERS = 5

sc = SpectralClustering(NB_CLUSTERS, affinity="precomputed", n_init=100, assign_labels="discretize")
corr = abs(df_sub_sub.corr())
clusters = sc.fit_predict(corr)
positions = np.argsort(clusters)


# %%

group = defaultdict(list)
for i, name in enumerate(list(corr.index)):
    group[clusters[i]].append(name)


# %%


reorg = corr.values[positions][positions]
corr.update(reorg)
# %%

# np.argsort(clusters)
# corr["cluster"] = clusters
# df = corr.sort_values(by=["cluster"]).drop(axis = 1, labels=["cluster"])

f, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(reorg, mask=np.triu(np.ones_like(reorg)))


# activité allelopathique : la plante pour elle mêem VS pharmaco : l'usage humain
# => deux catégories

"""Analysis module"""

# pylint: disable=unused-import
# %%
import csv
import time
import logging
import random

from collections import defaultdict
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from downloader import BASE_CLASS, get_classes, COMPOUNDS, PHARMACOLOGY, sorted_keys

logging.basicConfig()
logger = logging.getLogger("scopus_api")
logger.setLevel(logging.DEBUG)

INPUT_MATRIX = Path("results/activity.csv")
CSV_PARAMS = {"delimiter": ";", "quotechar": '"'}

CHEMO_MAP = get_classes(COMPOUNDS)
PHARM_MAP = get_classes(PHARMACOLOGY)
CHEMO_CLS_NB = sum(1 for cls in CHEMO_MAP.values() if cls == BASE_CLASS)
PHARM_CLS_NB = sum(1 for cls in PHARM_MAP.values() if cls == BASE_CLASS)


def load_results(filename: Path):
    """Loads chemical compounds / pharmacological activity matrice from SCOPUS"""
    logger.debug("load_results(%s)", filename)
    usecols = range(2, len(PHARM_MAP) + 2)
    full_matrix = np.loadtxt(filename, dtype=np.int32, delimiter=";", skiprows=2, encoding="utf-8", usecols=usecols)
    chemo_cls_nb = CHEMO_CLS_NB
    pharm_cls_nb = PHARM_CLS_NB

    logger.debug("%i chemo classes", chemo_cls_nb)
    logger.debug("%i pharm classes", pharm_cls_nb)
    # divide the full matrix into 4 quarter according to the
    # category of subjects : classes or base subject
    cls_cls = full_matrix[:chemo_cls_nb:, :pharm_cls_nb:]
    cls_sub = full_matrix[:chemo_cls_nb:, pharm_cls_nb::]
    sub_cls = full_matrix[chemo_cls_nb::, :pharm_cls_nb:]
    sub_sub = full_matrix[chemo_cls_nb::, pharm_cls_nb::]
    logger.info("dimensions of matrices %s %s %s %s", *map(lambda x: x.shape, [cls_cls, cls_sub, sub_cls, sub_sub]))
    # INFO:scopus_api:dimensions of matrices (5, 11) (5, 28) (53, 11) (53, 28)
    # 5*11 + 5*29 + 53*11 + 53*29 = 2320 = 58 * 40

    return [
        [cls_cls, cls_sub],
        [sub_cls, sub_sub],
    ]


chemo_cls = sorted_keys(CHEMO_MAP, base_only=True)
pharm_cls = sorted_keys(PHARM_MAP, base_only=True)
chemo_sub = sorted_keys(CHEMO_MAP, base_only=False)[CHEMO_CLS_NB::]
pharm_sub = sorted_keys(PHARM_MAP, base_only=False)[PHARM_CLS_NB::]


citations = load_results(INPUT_MATRIX)
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
plt.xticks(rotation=90 * 0.75)


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

# l'histogramme relatif
sns.set_theme(style="ticks")
# ici on peut normaliser, ici par ligne
df_sub_sub_n = df_sub_sub.apply(lambda x: 100 * x / df_sub_sub.sum(axis=1))
pdf_sub_sub_n = pd.melt(
    df_sub_sub_n.reset_index(), ignore_index=True, var_name="pharmacology", value_name="nb", id_vars="index"
)
pdf_sub_sub_n.rename(columns={"index": "compound"}, inplace=True)

# palette = sns.color_palette("husl", len(pharm_sub))

# %%
f, ax = plt.subplots(figsize=(12, 6))
sns.despine(f)
plt.xticks(rotation=90 * 0.75)
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
# on va randomiser les classes pharmaco
hue_order = pharm_sub.copy()
random.shuffle(hue_order)

# pylint: disable=unused-import
"""Download bibliographical information from Scopus"""

# %%

# import json
import requests
import asyncio
import logging
import ssl
import time
from os import environ
from pathlib import Path
from pprint import pprint

import aiohttp
import certifi
import pandas as pd
from dotenv import load_dotenv

from loader import Dataset, load_chemo_activities, write_chemo_activities

logging.basicConfig()
logger = logging.getLogger("chemo-diversity-scopus")
logger.setLevel(logging.DEBUG)

# take environment variables from .env.
# MUST DEFINE API_KEY with apy key from
load_dotenv()

# OUTPUT
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# API Scopus
API_KEY = {"X-ELS-APIKey": environ.get("API_KEY", "no-elsevier-api-key-defined")}
X_RATE_HEADERS = ["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"]
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

# output
OUTPUT_DIR = Path("results")
DEFAULT_INPUT = Path("data/activities.csv")
SAMPLE_INPUT = Path("data/samples.csv")
CSV_PARAMS = {"sep": ";", "quotechar": '"'}
ALT_SEP = "/"


def clausal_query(compounds, activities, pos_kw: list[str], neg_kw: list[str]):
    """Build a logical clause of the following form:

        (c_1 \/ ... \/ c_m)
     /\ (a_1 \/ ... \/ a_n)
     /\ (p_1 /\ ... /\ p_x)
     /\ (!n_1 /\ ... /\ !n_y)

    Where the dataset has m compounds and n activities,
    len(pos_kw) = x and len(neg_kw) = y.

    Keywords taht contain alternatives are normalized
    """

    def split_alts(string, op="OR"):
        base = f" {op} ".join(f"KEY({name})" for name in string.split(ALT_SEP))
        return f"({base})"

    # disjuncts = [slashes_to_or(kws) for kws in keywords]
    # KEY ( {disjunct1} ) AND KEY ( {disjunct2} )

    # compounds = df.index.get_level_values(1)
    # activities = df.columns.get_level_values(1)

    all_compounds_clause = " OR ".join(split_alts(compound) for compound in compounds)
    all_ativities_clause = " OR ".join(split_alts(activity) for activity in activities)
    positive_clause = " AND ".join(split_alts(kw) for kw in pos_kw)
    negative_clause = " OR ".join(split_alts(kw) for kw in neg_kw)

    # return all_compounds_clause, all_ativities_clause, positive_clause, negative_clause
    return f"({all_compounds_clause}) AND ({all_ativities_clause}) AND ({positive_clause}) AND NOT ({negative_clause})"


def wrap_scopus_query(string: str):
    return {"query": f'(DOCTYPE("ar")) AND ({string})', "count": 1}


def normalize_names(expr: str):
    return ALT_SEP.join(string.strip().lower() for string in expr.split(ALT_SEP))


async def query_scopus(query, delay=0):
    """SCOPUS query tool: return the number of article papers having two sets of keywords. Delay is in sec"""
    scopus_url = "https://api.elsevier.com/content/search/scopus"
    results_nb = -1
    
    await asyncio.sleep(delay)
    start_time = time.perf_counter()
    # logger.debug("query_scopus(%s) @%s", keywords, start_time)
    # logger.debug("           %s", query)
    try:
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            async with session.get(scopus_url, params=query, headers=API_KEY, ssl=SSL_CONTEXT) as resp:
                # logger.debug("X-RateLimit-Remaining=%s", resp.headers.get("X-RateLimit-Remaining", None))
                json = await resp.json()
                # print(json)
                results_nb = int(json["search-results"]["opensearch:totalResults"], 10)
                # logger.debug("query_scopus(%s): results_nb=%i", keywords, results_nb)
    except aiohttp.ClientError as err:
        logger.error(err)
        results_nb = -1
    finally:
        elapsed = time.perf_counter() - start_time
    return json

async def do(query):
    # loop = asyncio.get_event_loop()
    # # async with aiohttp.ClientSession(raise_for_status=True) as session:
    # main_task = loop.create_task(asyncio.sleep(2), name="main-queue")
    # results = loop.run_until_complete(main_task)
    # return results
    res = await asyncio.gather(query_scopus(query))
    # print(res)
    return res


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.debug("output dir is '%s'", OUTPUT_DIR.absolute())
    logger.debug("Scopus API key %s", API_KEY)

    # load data
    # https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
    df = pd.read_csv(SAMPLE_INPUT, index_col=[0, 1], header=[0, 1]).fillna(0)
    logger.debug("dataset %s read", SAMPLE_INPUT)

    # normalize strings
    df.index = pd.MultiIndex.from_tuples([tuple(normalize_names(x) for x in t) for t in df.index])
    df.columns = pd.MultiIndex.from_tuples([tuple(normalize_names(x) for x in t) for t in df.columns])

    # no names attribute to ensure read -> write is identity
    # , names = ["a-class", "compound"]
    # , names = ["a-class", "activity"]

    # remove first level on index
    # df = df.droplevel("C-Class", axis = 0)
    # df = df.droplevel("A-Class", axis = 1)

    # compounds = {y: x for (x, y) in df.index}
    # activities = {y: x for (x, y) in df.columns}

    # compounds_classes, compounds = df.index.levels
    # activities_classes, activities = df.columns.levels

    logger.info("%i compounds (with %i classes)", len(df.index.levels[1]), len(df.index.levels[0]))
    logger.info("%i activities (with %i classes)", len(df.columns.levels[1]), len(df.columns.levels[0]))

    cnf = clausal_query(
        df.index.get_level_values(1),
        df.columns.get_level_values(1),
        ["acridine"],
        ["cytotoxicity/toxicity"],
    )
    # print(cnf)
    print(wrap_scopus_query(cnf))



    res = asyncio.run(do(wrap_scopus_query(cnf)))
    print(res[0]["search-results"]["entry"][0])
    
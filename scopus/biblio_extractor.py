# pylint: disable=unused-import,anomalous-backslash-in-string
"""Generate queries and summarizes number of articles from bibliographical DB (e.g., Scopus)"""

# %%

import asyncio
import logging
import ssl
import time
from math import prod
from os import environ
from pathlib import Path
from posixpath import split
from pprint import pprint

import aiohttp
import certifi
import pandas as pd
from dotenv import load_dotenv

# take environment variables from .env.
# MUST DEFINE API_KEY with apy key from
logging.basicConfig()
logger = logging.getLogger("biblio.extractor")
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
INPUT_DATA = Path("data/activities.csv")
SAMPLE_DATA = Path("data/samples.csv")
TEST_DATA = Path("data/tests.csv")
CSV_PARAMS = {"sep": ";", "quotechar": '"'}
ALT_SEP = "/"


def load_data(file: str | Path):
    """loads data as a dataframe"""
    # https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
    df = pd.read_csv(file, index_col=[0, 1], header=[0, 1]).fillna(0)
    logger.debug("dataset %s read", file)

    def normalize_names(expr: str):
        """convenience tool"""
        return ALT_SEP.join(string.strip().lower() for string in expr.split(ALT_SEP))

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

    return df


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
        base = f" {op} ".join(f'KEY("{name}")' for name in string.split(ALT_SEP))
        return f"({base})"

    # disjuncts = [slashes_to_or(kws) for kws in keywords]
    # KEY ( {disjunct1} ) AND KEY ( {disjunct2} )

    # compounds = df.index.get_level_values(1)
    # activities = df.columns.get_level_values(1)

    all_compounds_clause = " OR ".join(split_alts(compound) for compound in compounds)
    all_ativities_clause = " OR ".join(split_alts(activity) for activity in activities)
    positive_clause = " AND ".join(split_alts(kw) for kw in pos_kw)
    negative_clause = " OR ".join(split_alts(kw) for kw in neg_kw)

    clauses = " AND ".join(
        f"({clause})" for clause in [all_compounds_clause, all_ativities_clause, positive_clause] if clause
    )

    if not (clauses):
        raise IndexError("at least one positive clause must be non-empty")

    if negative_clause:
        clauses += f" AND NOT ({negative_clause})"

    # return all_compounds_clause, all_ativities_clause, positive_clause, negative_clause
    # return f"({all_compounds_clause}) AND ({all_ativities_clause}) AND ({positive_clause}) AND NOT ({negative_clause})"

    return clauses


def wrap_scopus(string: str):
    if not (string):
        raise ValueError("string must be non-empty")
    return {"query": f'DOCTYPE("ar") AND {string}', "count": 1}


async def query_scopus(json_query, delay=0):
    """SCOPUS query tool: return the number of article papers having two sets of keywords. Delay is in sec"""
    scopus_url = "https://api.elsevier.com/content/search/scopus"
    results_nb = -1

    await asyncio.sleep(delay)
    start_time = time.perf_counter()
    # logger.debug("query_scopus(%s) @%s", keywords, start_time)
    # logger.debug("           %s", query)
    try:
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            async with session.get(scopus_url, params=json_query, headers=API_KEY, ssl=SSL_CONTEXT) as resp:
                logger.debug("X-RateLimit-Remaining=%s", resp.headers.get("X-RateLimit-Remaining", None))
                json = await resp.json()
                results_nb = int(json["search-results"]["opensearch:totalResults"], 10)
    except aiohttp.ClientError as err:
        logger.error(err)
        results_nb = -1
    finally:
        elapsed = time.perf_counter() - start_time
    logger.debug("query_scopus(): results_nb=%i in %f sec", results_nb, elapsed)
    return results_nb


async def do_async(query):
    # loop = asyncio.get_event_loop()
    # # async with aiohttp.ClientSession(raise_for_status=True) as session:
    # main_task = loop.create_task(asyncio.sleep(2), name="main-queue")
    # results = loop.run_until_complete(main_task)
    # return results
    res = await asyncio.gather(query_scopus(query))
    # print(res)
    return res


def generate_all_queries(data: pd.DataFrame):
    compounds = list(data.index.get_level_values(1))
    activities = list(data.columns.get_level_values(1))
    for compound in compounds:
        for activity in activities:
            # both the compound and the activity
            yield ([], [], [compound, activity], [])
            # the activity but not this compound (but at least one another in the domain)
            yield (compounds, [], [activity], [compound])
            # the compound but not this activity (but at least one another in the domain)
            yield ([], activities, [compound], [activity])
            # neither the compound nor the activity (but stil in the domain)
            yield (compounds, activities, [], [compound, activity])


    #rows margin sums
    for compound in compounds:
        yield ([], activities, [compound], [])
        yield (compounds, activities, [], [compound])
    #cols margin sums
    for activity in activities:
        yield (compounds, [], [activity], [])
        yield (compounds, activities, [], [activity])
    #total margin sum
    yield (compounds, activities, [], [])

if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("output dir is '%s'", OUTPUT_DIR.absolute())
    logger.info("Scopus API key %s", API_KEY)

    df = load_data(TEST_DATA)
    all_compounds = list(df.index.get_level_values(1))
    all_activities = list(df.columns.get_level_values(1))
    logger.debug("all compounds %s", all_compounds)
    logger.debug("all activities %s", all_activities)



    # print(df.columns.get_level_values(1))
    # print(cnf)
    # print(wrap_scopus_query(cnf))

    # res = asyncio.run(do_async(wrap_scopus_query(cnf)))
    # print(res)

    all_queries = list(generate_all_queries(df))
    # pprint(all_queries)
    logger.info("total number of queries: %i", len(all_queries))
    # print(res[0]["search-results"]["entry"][0])

    for query in all_queries:
        logger.info("query is %s", query[-2:])
        query_load = wrap_scopus(clausal_query(*query))
        # res = asyncio.run(do_async(query_load))


# df.loc[("shs","sociology"), ("computer science", "web")] = 12



# %%
    SELECTORS = ["w/", "w/o"]

    def extend_df(df:pd.DataFrame):
        mrows = pd.MultiIndex.from_tuples((cls, val, s) for (cls, val) in df.index for s in SELECTORS)
        mcols = pd.MultiIndex.from_tuples((cls, val, s) for (cls, val) in df.columns for s in SELECTORS)

        return pd.DataFrame(index=mrows, columns=mcols)


    df2=extend_df(df)

    df2.iloc[df2.index.get_level_values(2) == SELECTORS[0]]

    # margin_rows = pd.DataFrame(index=mrows, columns=pd.MultiIndex.from_tuples([("Σ","","w/"), ("Σ","","w/o")]))
    # pd.concat([df2, margin_rows])

    # %%

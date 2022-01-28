# pylint: disable=unused-import,anomalous-backslash-in-string
# pylint: disable=line-too-long
"""Generate queries and summarizes number of articles from bibliographical DB (e.g., Scopus)

The general idea is to have TWO disjoint finite sets of keywords such as :

- (chemical) coumpounds = KW1 = {acridine, triterpene, ...}
- (biological, pharmacological) activities = KW2 = {germination, cytotoxicity, ...}

Then to query an online bibligraphical service such as <https://api.elsevier.com/content/search/scopus> to find out how many papers have these keywords.
Each paper may have many keywords from the two sets, possibly none (open world hypothesis).

We want to analyse the dependencies between the two sets keywords using techniques like <https://en.wikipedia.org/wiki/Correspondence_analysis>
To do so, this program creates and fill specific kind of contingency table such as follows :

                germination germination cytotoxicity    cytotoxicity
                w/          w/          w/              w/o
acridine    w/  U_11        V_11        U_12            V_12
acridine    w/o X_11        Y_11        X_12            Y_12
triterpene  w/  U_21        V_21        U_22            V_22
triterpene  w/o X_21        Y_21        X_22            Y_22

Where for each couple (kw_i, kw_j) in KW1xKW2, the submatrix [U,V][X,Y] gives :

- U = the number of papers that have both (kw1, kw2) as keywords
- V = the number of papers that have kw1 but NOT kw2 as keywords
- X = the number of papers that have kw2 but NOT kw1 as keywords
- Y = the number of papers that have NEITHER kw1 NOR kw2 as keywords

We restrict the analysis to the domain D, which is the set of paper that have at least one keyword in KW1 and at least one in KW2.
So, by contruction each submatrix [U,V][X,Y] is such that U + V + X + Y  = |D|
"""
# pylint: enable=line-too-long

# %%

import asyncio
from itertools import product
import logging
import ssl
import time
from os import environ
from pathlib import Path

import aiohttp
import certifi
import pandas as pd
from dotenv import load_dotenv

# take environment variables from .env.
# MUST DEFINE API_KEY with apy key from
load_dotenv()

if __name__ == "__main__":
    logging.basicConfig()

logger = logging.getLogger(f"CHEMOTAXO.{__name__}")

# Scopus API
API_KEY = {"X-ELS-APIKey": environ.get("API_KEY", "no-elsevier-api-key-defined")}
X_RATE_HEADERS = ["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"]
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

# Output
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Input
INPUT_DATA = Path("data/activities.csv")
SAMPLE_DATA = Path("data/samples.csv")
TEST_DATA = Path("data/tests.csv")

# I/O configuration
CSV_PARAMS = {"sep": ";", "quotechar": '"'}
ALT_SEP = "/"
SELECTORS = ["w/", "w/o"]
MARGIN_SYMB = "Σ"
CLASS_SYMB = "*"


def load_data(file: str | Path):
    """loads a CSV dataset as a dataframe"""
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

    def split_alts(string, operator="OR"):
        base = f" {operator} ".join(f'KEY("{name}")' for name in string.split(ALT_SEP))
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

    if not clauses:
        raise IndexError("at least one positive clause must be non-empty")

    if negative_clause:
        clauses += f" AND NOT ({negative_clause})"

    return clauses


def wrap_scopus(string: str):
    """Wraps a string query into an object to be sent as JSON over Scopus API"""
    if not string:
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
    """Launch aysync job"""
    # loop = asyncio.get_event_loop()
    # # async with aiohttp.ClientSession(raise_for_status=True) as session:
    # main_task = loop.create_task(asyncio.sleep(2), name="main-queue")
    # results = loop.run_until_complete(main_task)
    # return results
    res = await asyncio.gather(query_scopus(query))
    # print(res)
    return res


def generate_all_queries(data: pd.DataFrame, with_margin=False):
    """Generate all values to fill the contengency table"""
    compounds = list(data.index.get_level_values(1))
    activities = list(data.columns.get_level_values(1))

    # the main content : 4 x |KW1| x |KW2| cells
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

    # adds extra rows/columns for marginal sums (an extra row and an extra column for total)
    # this should add 4 x (|KW1| + |KW2| + 1) but we exclude 2 + 2 + 3 degenerated combinations which always are 0
    if with_margin:
        # rows margin sums, -2 always 0
        for compound in compounds:
            yield ([], activities, [compound], [])
            yield (compounds, activities, [], [compound])
        # cols margin sums, -2 always 0
        for activity in activities:
            yield (compounds, [], [activity], [])
            yield (compounds, activities, [], [activity])
        # total margin sum, -3 always 0
        yield (compounds, activities, [], [])


def extend_df(df: pd.DataFrame, with_margin=False) -> pd.DataFrame:
    """Add extra indexes as last level of rows and columns.

    Index and columns are multi-level indexes. We duplicate each key to have
    an extra [w/, w/o] index level at the finest level.

    In the end, the orginal KW1 x KW2 matrix is transformed to a 4 x KW1 x KW2 one
    each original celle [m] being now a 2x2 submatrix [U, V][X, Y]

    If margin are added, a  4 x (KW1 + 1) x (KW2 + 1) is constructed
    """
    df2 = pd.DataFrame().reindex_like(df)

    if with_margin:
        margin_row = pd.DataFrame(index=pd.MultiIndex.from_tuples([(CLASS_SYMB, MARGIN_SYMB)]), columns=df.columns)
        df2 = df2.append(margin_row)
        df2[(CLASS_SYMB, MARGIN_SYMB)] = None

    extended_rows = pd.MultiIndex.from_tuples((cls, val, s) for (cls, val) in df2.index for s in SELECTORS)
    extended_cols = pd.MultiIndex.from_tuples((cls, val, s) for (cls, val) in df2.columns for s in SELECTORS)

    return pd.DataFrame(index=extended_rows, columns=extended_cols)

    # df2 = pd.DataFrame(index=mrows, columns=mcols)

    # if with_margin:
    #     margin_rows = pd.DataFrame(index=mrows, columns=pd.MultiIndex.from_tuples(product(["Σ"], SELECTORS)))
    #     margin_cols = pd.DataFrame(index=mcols, columns=pd.MultiIndex.from_tuples(product(["Σ"], SELECTORS)))
    #     df2 = pd.concat([df2, margin_rows])

    # return df2


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("output dir is '%s'", OUTPUT_DIR.absolute())
    logger.info("Scopus API key %s", API_KEY)

    dataset = load_data(TEST_DATA)
    all_compounds = list(dataset.index.get_level_values(1))
    all_activities = list(dataset.columns.get_level_values(1))
    logger.debug("all compounds %s", all_compounds)
    logger.debug("all activities %s", all_activities)

    # print(df.columns.get_level_values(1))
    # print(cnf)
    # print(wrap_scopus_query(cnf))

    # res = asyncio.run(do_async(wrap_scopus_query(cnf)))
    # print(res)

    all_queries = list(generate_all_queries(dataset))
    # pprint(all_queries)
    logger.info("total number of queries: %i", len(all_queries))
    # print(res[0]["search-results"]["entry"][0])

    for query in all_queries:
        logger.info("query is %s", query[-2:])
        query_load = wrap_scopus(clausal_query(*query))
        # res = asyncio.run(do_async(query_load))

    # df.loc[("shs","sociology"), ("computer science", "web")] = 12

    # %%
    dataset2 = extend_df(dataset)
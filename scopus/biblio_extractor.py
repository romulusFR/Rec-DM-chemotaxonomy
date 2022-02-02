# pylint: disable=unused-import
# pylint: disable=anomalous-backslash-in-string
# pylint: disable=line-too-long
"""Generate queries and summarizes number of articles from bibliographical DB (e.g., Scopus)

The general idea is to have TWO (disjoint) finite sets of keywordss, for example:

- KW1 = a set of (chemical) coumpounds = {acridine, triterpene, ...}
- KW2 = a set of (biological, pharmacological) activities = {germination, cytotoxicity, ...}

The program queries an online bibligraphical service such as <https://api.elsevier.com/content/search/scopus>
to find out how many papers have these keywords.
Note that, each paper may have many keywords from the two sets, possibly none (open world hypothesis).

We want to analyse the dependencies between the two sets keywords using techniques like <https://en.wikipedia.org/wiki/Correspondence_analysis>
To do so, this program creates and fill specific kind of contingency table such as follows :

                germination germination cytotoxicity    cytotoxicity
                w/o         w/          w/o             w/
acridine    w/o U_11        V_11        U_12            V_12
acridine    w/  X_11        Y_11        X_12            Y_12
triterpene  w/o U_21        V_21        U_22            V_22
triterpene  w/  X_21        Y_21        X_22            Y_22

Where for each couple (kw_i, kw_j) in KW1xKW2, the submatrix [U,V][X,Y] is a kind of confusion matrix where:

- U = (False, False) is the number of papers that have NEITHER kw1 NOR kw2 as keywords
- V = (False, True)  is the number of papers that have kw2 but NOT kw1 as keywords
- X = (True, False)  is the number of papers that have kw1 but NOT kw2 as keywords
- Y = (True, True)   is the number of papers that have BOTH kw1 AND kw2 as keywords


We avoid the open world hypothesis by restricting the analysis to the paper in the domain D,
which is the set of paper that have at least one keyword in KW1 and at least one in KW2.
By construction:
- U + V and X + Y are constants for each kw1 (whatever the choice of kw2)
- U + X and V + Y are constants for each kw2 (whatever the choice of kw1)
Moreover each confusion matrix [U,V][X,Y] is such that U + V + X + Y  = |D|.

"""
# pylint: enable=line-too-long

# %%

import asyncio
import logging
import ssl
import time
from collections import defaultdict
from functools import partial, wraps
from itertools import product
from os import environ
from pathlib import Path
from random import randint, sample

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

# Web requests / API
API_KEY = {"X-ELS-APIKey": environ.get("API_KEY", "no-elsevier-api-key-defined")}
X_RATE_HEADERS = ["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"]
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

# Input samples
INPUT_DATA = Path("data/activities.csv")
SAMPLE_DATA = Path("data/samples.csv")
TEST_DATA = Path("data/tests.csv")

# I/O and string configuration
CSV_PARAMS = {"sep": ";", "quotechar": '"'}
ALT_SEP = "/"
SELECTORS = ["w/o", "w/"]  # ordered as bools
MARGIN_SYMB = "Σ"
CLASS_SYMB = "*"

# Default parameters
DEFAULT_PARALLEL_WORKERS = 8  # number of parallel jobs
DEFAULT_WORKER_DELAY = 1.0  # at most one req / sec
DEFAULT_SAMPLES = None  # no sampling

# Typing
# a keyword is a fully index row or column identifier
Keyword = tuple[str, str]
# an aliases for queries : KW1, KW2, POS_KW, NEG_KW, KIND
# where KIND defines the combination among {w/o, w/}x{w/o, w/}
# that is, a celle of the confusion matrix
Query = tuple[list[Keyword], list[Keyword], list[Keyword], list[Keyword], tuple[bool, bool]]


def load_data(filename: str | Path):
    """loads a CSV dataset as a dataframe"""
    # https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
    df: pd.DataFrame = pd.read_csv(filename, index_col=[0, 1], header=[0, 1]).fillna(0)
    logger.debug("dataset %s read", filename)

    def normalize_names(expr: str):
        """convenience tool"""
        return ALT_SEP.join(string.strip().lower() for string in expr.split(ALT_SEP))

    # normalize strings
    df.index = pd.MultiIndex.from_tuples([tuple(normalize_names(x) for x in t) for t in df.index])
    df.columns = pd.MultiIndex.from_tuples([tuple(normalize_names(x) for x in t) for t in df.columns])

    logger.info("%i compounds (with %i classes)", len(df.index.levels[1]), len(df.index.levels[0]))
    logger.info("%i activities (with %i classes)", len(df.columns.levels[1]), len(df.columns.levels[0]))

    return df


def extend_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add extra indexes as last level of rows and columns.

    Index and columns are multi-level indexes. We duplicate each key to have
    an extra [w/, w/o] index level at the finest level.

    In the end, the orginal KW1 x KW2 matrix is transformed to a 4 x KW1 x KW2 one
    each original celle [m] being now a 2x2 submatrix [U, V][X, Y]

    OBSOLETE : if margin are added, a  4 x (KW1 + 1) x (KW2 + 1) is constructed
    """
    df2 = pd.DataFrame().reindex_like(df)

    # if with_margin:

    # margin_row = pd.DataFrame(index=pd.MultiIndex.from_tuples([(CLASS_SYMB, MARGIN_SYMB)]), columns=df.columns)
    # df2 = pd.concat([df2, margin_row], axis=0)
    # margin_col = pd.DataFrame(index=df.columns, columns=pd.MultiIndex.from_tuples([(CLASS_SYMB, MARGIN_SYMB)]))
    # df2 = pd.concat([df2, margin_col], axis=1)

    # df2 = df2.append(margin_row)
    # df2[(CLASS_SYMB, MARGIN_SYMB)] = None
    # df2[(CLASS_SYMB, MARGIN_SYMB)] = df2[(CLASS_SYMB, MARGIN_SYMB)].astype(int)

    extended_rows = pd.MultiIndex.from_tuples((cls, val, s) for (cls, val) in df2.index for s in SELECTORS)
    extended_cols = pd.MultiIndex.from_tuples((cls, val, s) for (cls, val) in df2.columns for s in SELECTORS)

    # https://pandas.pydata.org/pandas-docs/stable/user_guide/integer_na.html
    extended_df = pd.DataFrame(index=extended_rows, columns=extended_cols).astype("Int64")
    return extended_df


def clausal_query(query: Query) -> str:
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

    compounds, activities, pos_kw, neg_kw, _ = query

    # we only keep the last item [-1] of keywords, i.e., we discard their classes in queries
    all_compounds_clause = " OR ".join(split_alts(compound[-1]) for compound in compounds)
    all_ativities_clause = " OR ".join(split_alts(activity[-1]) for activity in activities)
    positive_clause = " AND ".join(split_alts(kw[-1]) for kw in pos_kw)
    negative_clause = " OR ".join(split_alts(kw[-1]) for kw in neg_kw)

    clauses = " AND ".join(
        f"({clause})" for clause in [all_compounds_clause, all_ativities_clause, positive_clause] if clause
    )

    if not clauses:
        raise IndexError("at least one positive clause must be non-empty")

    if negative_clause:
        clauses += f" AND NOT ({negative_clause})"

    return clauses


async def fake_search(_, query: Query, *, delay=0):
    """Fake query tool without network, for test purpose"""
    await asyncio.sleep(delay)
    start_time = time.perf_counter()
    clause = clausal_query(query)
    logger.debug("fake_search(%s)", query[-3:])
    logger.debug("               %s", clause)
    results_nb = randint(1, 10000)
    await asyncio.sleep(randint(1, 1000) / 1000)
    elapsed = time.perf_counter() - start_time
    return results_nb, elapsed


async def httpbin_search(session, query: Query, *, delay=0, error_rate=10):
    """Fake query tool WITH network on httpbin, for test purpose"""
    # simule 1% d'erreur
    if randint(1, 100) <= error_rate:
        url = "http://httpbin.org/status/429"
    else:
        url = "http://httpbin.org/anything"
    data = {"answer": randint(1, 10000)}
    results_nb = None
    await asyncio.sleep(delay)
    start_time = time.perf_counter()
    logger.debug("httpbin_search(%s)", query[-3:])
    json_query = wrap_scopus(clausal_query(query))

    try:
        # async with aiohttp.ClientSession(raise_for_status=True) as session:
        async with session.get(url, params=json_query, data=data, ssl=SSL_CONTEXT) as resp:
            json = await resp.json()
            results_nb = int(json["form"]["answer"])
    except aiohttp.ClientResponseError as err:
        logger.warning("aiohttp.ClientResponseError #%i: %s", err.status, err.message)
    finally:
        elapsed = time.perf_counter() - start_time
    logger.debug("httpbin_search(%s)=%i in %f sec", query[-3:], results_nb, elapsed)
    return results_nb, elapsed


def wrap_scopus(string: str):
    """Wraps a string query into an object to be sent as JSON over Scopus API"""
    if not string:
        raise ValueError("string must be non-empty")
    return {"query": f'DOCTYPE("ar") AND {string}', "count": 1}


async def scopus_search(session, query: Query, *, delay=0):
    """SCOPUS query tool: return the number of article papers having two sets of keywords. Delay is in sec"""
    scopus_url = "https://api.elsevier.com/content/search/scopus"
    results_nb = None

    await asyncio.sleep(delay)
    start_time = time.perf_counter()
    logger.debug("scopus_search(%s)", query[-3:])
    json_query = wrap_scopus(clausal_query(query))
    try:
        # async with aiohttp.ClientSession(raise_for_status=True) as session:
        async with session.get(scopus_url, params=json_query, headers=API_KEY, ssl=SSL_CONTEXT) as resp:
            logger.debug("X-RateLimit-Remaining=%s", resp.headers.get("X-RateLimit-Remaining", None))
            json = await resp.json()
            results_nb = int(json["search-results"]["opensearch:totalResults"], 10)
    except aiohttp.ClientResponseError as err:
        logger.warning("aiohttp.ClientResponseError #%i: %s", err.status, err.message)
    finally:
        elapsed = time.perf_counter() - start_time
    logger.debug("scopus_search(%s)=%s in %f sec", query[-3:], results_nb, elapsed)
    return results_nb, elapsed


# query modes : one fake, one fake over network, one true
SEARCH_MODES = {"scopus": scopus_search, "httpbin": httpbin_search, "fake": fake_search}
DEFAULT_SEARCH_MODE = "fake"


def generate_all_queries(data: pd.DataFrame, *, with_margin=False):
    """Generate all values to fill the contengency table"""
    # compounds = list(data.index.get_level_values(1))
    # activities = list(data.columns.get_level_values(1))

    compounds = data.index.to_list()
    activities = data.columns.to_list()

    # the main content : 4 x |KW1| x |KW2| cells
    for compound in compounds:
        for activity in activities:
            # both the compound and the activity
            yield ([], [], [compound, activity], [], (True, True))
            # the activity but not this compound (but at least one another in the domain)
            yield (compounds, [], [activity], [compound], (False, True))
            # the compound but not this activity (but at least one another in the domain)
            yield ([], activities, [compound], [activity], (True, False))
            # neither the compound nor the activity (but stil in the domain)
            yield (compounds, activities, [], [compound, activity], (False, False))

    # adds extra rows/columns for marginal sums (an extra row and an extra column for total)
    # this should add 4 x (|KW1| + |KW2| + 1) but we exclude 2 + 2 + 3 degenerated combinations which always are 0
    if with_margin:
        # rows margin sums, -2 always 0
        for compound in compounds:
            yield ([], activities, [compound], [], (True, None))
            yield (compounds, activities, [], [compound], (False, None))
        # cols margin sums, -2 always 0
        for activity in activities:
            yield (compounds, [], [activity], [], (None, True))
            yield (compounds, activities, [], [activity], (None, False))
        # total margin sum, -3 always 0
        yield (compounds, activities, [], [], (None, None))


async def execute_job(session, queue, results_df, task_factory, *, worker_delay=1, name=None):
    """A (parallel) consumer that send a query to scopus and then add result to a dataframe"""
    jobs_done = 0
    jobs_retried = 0
    try:
        while not queue.empty():
            query = await queue.get()
            nb_results, duration = await task_factory(session, query, delay=0)

            logger.info("execute_job(id=%s) got %s from job %s after %f", name, nb_results, query[-3:], duration)
            pos_kw, neg_kw, kind = query[-3:]
            if kind == (True, True):
                results_df.loc[(*pos_kw[0], SELECTORS[True]), (*pos_kw[1], SELECTORS[True])] = nb_results
            elif kind == (True, False):
                results_df.loc[(*pos_kw[0], SELECTORS[True]), (*neg_kw[0], SELECTORS[False])] = nb_results
            elif kind == (False, True):
                results_df.loc[(*neg_kw[0], SELECTORS[False]), (*pos_kw[0], SELECTORS[True])] = nb_results
            elif kind == (False, False):
                results_df.loc[(*neg_kw[0], SELECTORS[False]), (*neg_kw[1], SELECTORS[False])] = nb_results
            elif kind == (True, None):
                results_df.loc[(*pos_kw[0], SELECTORS[True]), (CLASS_SYMB, MARGIN_SYMB, SELECTORS[True])] = nb_results
            elif kind == (False, None):
                results_df.loc[(*neg_kw[0], SELECTORS[False]), (CLASS_SYMB, MARGIN_SYMB, SELECTORS[True])] = nb_results
            elif kind == (None, True):
                results_df.loc[(CLASS_SYMB, MARGIN_SYMB, SELECTORS[True]), (*pos_kw[0], SELECTORS[True])] = nb_results
            elif kind == (None, False):
                results_df.loc[(CLASS_SYMB, MARGIN_SYMB, SELECTORS[True]), (*neg_kw[0], SELECTORS[False])] = nb_results
            elif kind == (None, None):
                results_df.loc[
                    (CLASS_SYMB, MARGIN_SYMB, SELECTORS[True]), (CLASS_SYMB, MARGIN_SYMB, SELECTORS[True])
                ] = nb_results
            else:
                # raise ValueError(f"{len(pos_kw) = }, {len(neg_kw) = } for {kind = } should not arise")
                logger.error(
                    "execute_job(id=%s): len(pos_kw) = %i, len(neg_kw) = %i should not arise for kind = %s",
                    name,
                    len(pos_kw),
                    len(neg_kw),
                    kind,
                )
            queue.task_done()
            jobs_done += 1

            # add the same query again in the job queue to retry it
            if nb_results is None:
                await queue.put(query)
                jobs_retried += 1
                logger.error("execute_job(id=%s) added back %s to the queue", name, query[-3:])

            await asyncio.sleep(max(worker_delay - duration, 0))
    except asyncio.CancelledError:
        logger.debug("execute_job() task %s received cancel, done %i jobs, retried %i", name, jobs_done, jobs_retried)

    return jobs_done, jobs_retried


async def jobs_spawner(df: pd.DataFrame, *, task_factory, with_margin, parallel_workers, worker_delay, samples):
    """Create tasks in a queue which is emptied in parallele ensuring at most MAX_REQ_BY_SEC requests per second"""
    jobs_queue: asyncio.Queue = asyncio.Queue()
    logger.info("jobs_spawner(): task_factory=%s, parallel_workers=%i", task_factory.__name__, parallel_workers)

    # generate all queries put them into the queue
    all_queries = list(generate_all_queries(df, with_margin=with_margin))
    if samples is not None:
        all_queries = sample(all_queries, samples)
    for query in all_queries:
        await jobs_queue.put(query)
        logger.debug("jobs_spawner() added query=%s", query[-3:])

    logger.debug("jobs_spawner() added %i queries", len(all_queries))

    # MAX_REQ_BY_SEC consummers that run in parallel and that can fire at most
    # one request per second
    consumer_tasks = []
    result_df = extend_df(df)
    async with aiohttp.ClientSession(raise_for_status=True) as session:

        # on lance tous les exécuteurs de requêtes
        consumer_tasks = [
            asyncio.create_task(
                execute_job(session, jobs_queue, result_df, task_factory, worker_delay=worker_delay, name=i),
                name=f"consumer-{i}",
            )
            for i in range(1, parallel_workers + 1)
        ]

        logger.info("jobs_spawner() %i consumer tasks created", len(consumer_tasks))

        # on attend que tout soit traité, après que tout soit généré
        await jobs_queue.join()
        logger.debug("jobs_spawner() job queue is empty")
        # OBSOLETE
        # stop all consumer stuck waiting job from the queue if any
        # for consumer in consumer_tasks:
        #     consumer.cancel()
        #     logger.debug("jobs_spawner() %s stopped", consumer.get_name())
        jobs_done = await asyncio.gather(*consumer_tasks, return_exceptions=True)
        logger.info("jobs_spawner() nb of jobs done by each worker %s", jobs_done)

    return result_df


def launcher(
    df: pd.DataFrame,
    *,
    task_factory=SEARCH_MODES[DEFAULT_SEARCH_MODE],
    with_margin=False,
    parallel_workers=DEFAULT_PARALLEL_WORKERS,
    worker_delay=DEFAULT_WORKER_DELAY,
    samples=None,
):
    """Launch the batch of downloads: a simple wrapper around jobs_spawner"""
    launch_start_time = time.perf_counter()
    logger.info("launcher() starting asynchronous jobs")
    logger.info("launcher() launching all jobs (producer and consumers)")
    results_df = asyncio.run(
        jobs_spawner(
            df,
            parallel_workers=parallel_workers,
            task_factory=task_factory,
            worker_delay=worker_delay,
            with_margin=with_margin,
            samples=samples,
        )
    )

    total_time = time.perf_counter() - launch_start_time
    logger.info("launcher() all jobs done in %fs", total_time)

    return results_df.astype("Int64")


# %%
if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    logger.info("__main__ Scopus API key %s", API_KEY)

    dataset = load_data(TEST_DATA)
    all_compounds = list(dataset.index.get_level_values(1))
    all_activities = list(dataset.columns.get_level_values(1))
    logger.debug("__main__ all compounds %s", all_compounds)
    logger.debug("__main__ all activities %s", all_activities)

    # task_factory = partial(httpbin_search, error_rate=50)
    # task_factory.__name__ = httpbin_search.__name__
    results = launcher(dataset)
    print(results)
    print(results.info())

    # print(df.columns.get_level_values(1))
    # print(cnf)
    # print(wrap_scopus_query(cnf))

    # res = asyncio.run(do_async(wrap_scopus_query(cnf)))
    # print(res)

    # all_queries = list(generate_all_queries(dataset, with_margin=True))
    # # pprint(all_queries)
    # logger.info("total number of queries: %i", len(all_queries))
    # # print(res[0]["search-results"]["entry"][0])

    # for a_query in all_queries:  # [-2:]
    #     # logger.debug("query is %s", a_query[-2:])
    #     # query_load = wrap_scopus(clausal_query(*query))
    #     res = asyncio.run(do_async(a_query))
    #     print(res)

    # df.loc[("shs","sociology"), ("computer science", "web")] = 12

    # %%
    # results = extend_df(dataset)
    # results.loc[("shs", "sociology"),("computer science", "web")]
    # results.loc[("shs", "sociology", SELECTORS[0]),("computer science", "web", SELECTORS[0])] = 42
    # results

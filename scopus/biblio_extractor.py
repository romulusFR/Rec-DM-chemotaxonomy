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

# TODO : gérer les tâches qui plantent

import asyncio
from functools import wraps
from itertools import product
import logging
import ssl
from random import randint
import time
from os import environ
from pathlib import Path
from collections import defaultdict

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
# ordered as bools
SELECTORS = ["w/o", "w/"]
MARGIN_SYMB = "Σ"
CLASS_SYMB = "*"

# a keyword is a fully index row or column identifier
Keyword = tuple[str, str]
# an aliases for queries : KW1, KW2, POS_KW, NEG_KW
Query = tuple[list[Keyword], list[Keyword], list[Keyword], list[Keyword], tuple[bool, bool]]


def load_data(file: str | Path):
    """loads a CSV dataset as a dataframe"""
    # https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
    df: pd.DataFrame = pd.read_csv(file, index_col=[0, 1], header=[0, 1]).fillna(0)
    logger.debug("dataset %s read", file)

    def normalize_names(expr: str):
        """convenience tool"""
        return ALT_SEP.join(string.strip().lower() for string in expr.split(ALT_SEP))

    # normalize strings
    df.index = pd.MultiIndex.from_tuples([tuple(normalize_names(x) for x in t) for t in df.index])
    df.columns = pd.MultiIndex.from_tuples([tuple(normalize_names(x) for x in t) for t in df.columns])

    logger.info("%i compounds (with %i classes)", len(df.index.levels[1]), len(df.index.levels[0]))
    logger.info("%i activities (with %i classes)", len(df.columns.levels[1]), len(df.columns.levels[0]))

    return df


def extend_df(df: pd.DataFrame, *, with_margin=False) -> pd.DataFrame:
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

        margin_col = pd.DataFrame(columns=pd.MultiIndex.from_tuples([(CLASS_SYMB, MARGIN_SYMB)]), index=df.columns)

        df2 = pd.concat([df2, margin_col], axis=1)
        df2 = pd.concat([df2, margin_row], axis=0)


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

    compounds, activities, pos_kw, neg_kw, kind = query

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


async def httpbin_search(session, query: Query, *, delay=0, error_rate=1):
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
    logger.debug("scopus_search(%s)=%i in %f sec", query[-3:], results_nb, elapsed)
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


async def create_all_job(queue, df, *, with_margin=False):
    """Adds jobs (queries) into the job queue"""
    # generate all queries
    all_queries = list(generate_all_queries(df, with_margin=with_margin))

    # put them into the queue with a job number
    for query in all_queries:
        await queue.put(query)
        logger.debug("create_all_tasks() added query=%s", query[-3:])

    return len(all_queries)


# async def do_async(query: Query):
#     """Launch aysync job"""
#     # loop = asyncio.get_event_loop()
#     # # async with aiohttp.ClientSession(raise_for_status=True) as session:
#     # main_task = loop.create_task(asyncio.sleep(2), name="main-queue")
#     # results = loop.run_until_complete(main_task)
#     # return results

#     # res = await asyncio.gather(scopus_search(query))
#     # return res
#     async with aiohttp.ClientSession(raise_for_status=True) as session:
#         results = await asyncio.gather(httpbin_search(session, query, error_rate=50))
#     return results


async def execute_job(session, queue, results_df, task_factory, *, worker_delay=1, name=None):
    """A (parallel) consumer that send a query to scopus and then add result to a dict"""
    jobs_done = 0
    try:
        while True:
            query = await queue.get()
            result = await task_factory(session, query, delay=0)
            nb_results, duration = result
            # TODO :  reprise sur erreur, when nb_results is None
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
                logger.error(f"{len(pos_kw) = }, {len(neg_kw) = } for {kind = } should not arise")
            queue.task_done()
            jobs_done += 1
            await asyncio.sleep(max(worker_delay - duration, 0))
    except asyncio.CancelledError:
        logger.debug("execute_job() task %s received cancel, done %i jobs", name, jobs_done)
    return jobs_done


async def jobs_spawner(df: pd.DataFrame, task_factory, *, parallel_workers, worker_delay, with_margin):
    """Create tasks in a queue which is emptied in parallele ensuring at most MAX_REQ_BY_SEC requests per second"""
    jobs_queue: asyncio.Queue = asyncio.Queue()
    # res_dict: defaultdict = defaultdict(dict)
    logger.info("jobs_spawner(_, %s, %i, _, _)", task_factory.__name__, parallel_workers)

    # ONE producer task to fill the queue
    producer_task = asyncio.create_task(create_all_job(jobs_queue, df, with_margin=with_margin), name="producer")

    # on lance le producteur qui peuple la queue
    [nb_queries] = await asyncio.gather(producer_task)
    logger.info("jobs_spawner() producer added %i queries in the job queue", nb_queries)

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
        # stop all consumer stuck waiting job from the queue if any
        for consumer in consumer_tasks:
            consumer.cancel()
            logger.debug("jobs_spawner() %s stopped", consumer.get_name())
        jobs_done = await asyncio.gather(*consumer_tasks)
        logger.info("jobs_spawner() nb of jobs done by each worker %s", jobs_done)

    # pending = asyncio.all_tasks()
    # logger.debug(pending)

    return result_df


DEFAULT_PARALLEL_WORKERS = 8
DEFAULT_WORKER_DELAY = 1.0
DEFAULT_SAMPLES = None


def launcher(df: pd.DataFrame, *, with_margin=False):
    """Launch the batch of downloads"""
    # compounds = dataset.compounds.keys()
    # activities = dataset.activities.keys()
    # queries = list(product(compounds, activities))

    # if samples is not None:
    # queries = sample(queries, samples)  # [(c,) for c in compounds]

    # logger.info("%i compounds X %i pharmacology = %i", len(compounds), len(activities), len(queries))

    launch_start_time = time.perf_counter()
    logger.info("launcher() starting asynchronous jobs")
    task_factory = SEARCH_MODES.get(DEFAULT_SEARCH_MODE, SEARCH_MODES[DEFAULT_SEARCH_MODE])

    # correction bug scopus
    # results = asyncio.run(main_queue(queries, mode=mode))
    # TODO : supprimer loop
    loop = asyncio.get_event_loop()
    main_task = loop.create_task(
        jobs_spawner(
            df,
            parallel_workers=DEFAULT_PARALLEL_WORKERS,
            task_factory=task_factory,
            worker_delay=DEFAULT_WORKER_DELAY,
            with_margin=with_margin,
        ),
        name="main-queue",
    )
    logger.info("launcher() launching all jobs (producer and consumers)")
    results_df = loop.run_until_complete(main_task)

    # dataset.data = results

    total_time = time.perf_counter() - launch_start_time
    logger.info("launcher() all jobs done in %fs", total_time)

    # if not no_write:
    #     now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #     output_filename = OUTPUT_DIR / f"activities_{now}.csv"
    #     write_chemo_activities(output_filename, dataset)
    #     logger.info("WRITTEN %s", output_filename)

    return results_df


# %%
if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("__main__ output dir is '%s'", OUTPUT_DIR.absolute())
    logger.info("__main__ Scopus API key %s", API_KEY)

    dataset = load_data(TEST_DATA)
    all_compounds = list(dataset.index.get_level_values(1))
    all_activities = list(dataset.columns.get_level_values(1))
    logger.debug("__main__ all compounds %s", all_compounds)
    logger.debug("__main__ all activities %s", all_activities)

    results = launcher(dataset, with_margin=True)
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

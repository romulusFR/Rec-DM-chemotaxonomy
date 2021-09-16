"""test de queries async à scopus via aoihttp"""
# pylint: disable=unused-import
# %%

import argparse
import asyncio
import csv
import logging
import ssl
import time
from collections import defaultdict
from datetime import datetime
from itertools import product
from pathlib import Path
from pprint import pprint
from random import randint, sample
from typing import Tuple

import aiohttp
import certifi
import numpy as np

logging.basicConfig()
logger = logging.getLogger("scopus_api")
logger.setLevel(logging.DEBUG)

# https://www.twilio.com/blog/asynchronous-http-requests-in-python-with-aiohttp
# https://stackoverflow.com/questions/48682147/aiohttp-rate-limiting-parallel-requests
# https://docs.aiohttp.org/en/stable/client.html
# https://docs.aiohttp.org/en/stable/client_advanced.html


# SAMPLES / TESTS
CHEMISTRY = ["alkaloid", "polyphenol", "coumarin"]
ACTIVITIES = ["antiinflammatory", "anticoagulant", "cancer"]
RESULTS = {"acridine": {"anticancer": "2790"}, "pyridine": {"toxicant": "1904"}, "tetracycline": {"repulsive": "4598"}}
QUERIES = list(product(CHEMISTRY, ACTIVITIES))
ERROR_RATE = 1  # (in %)

# DATASETS
COMPOUNDS = Path("data") / "compounds.csv"
PHARMACOLOGY = Path("data") / "pharmacology.csv"
BASE_CLASS = "*"

# OUTPUT
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PARAMS = {"delimiter": ";", "quotechar": '"'}

# API Scopus
API_KEY = {"X-ELS-APIKey": "7047b3a8cf46d922d5d5ca71ff531b7d"}
X_RATE_HEADERS = ["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"]
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

# DEFAULT VALUES
DEFAULT_WORKERS = 8
DEFAULT_DELAY_PER_WORKER = 1.0
DEFAULT_SAMPLES = None
DEFAULT_MODE = "mock"


def get_classes(filename):
    """Loads classes from a csv file with format (parent, child) and return a dict with ALL names"""
    logger.debug("get_classes(%s)", filename)
    classes = {}
    with open(filename, encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, **CSV_PARAMS)
        for (parent, child) in reader:
            classes[standardize(child)] = standardize(parent)

    # add parents classes to the dict, with a default parent for each one
    for value in set(classes.values()):
        classes[value] = BASE_CLASS
    return classes


# NOTE : rows et cols sont surtout pour l'ordreet les cases vides si besoin
def write_results(res_dict: dict[Tuple[str, str], Tuple[int, float]], rows, cols, row_classes, col_classes, filename):
    """Store result dict"""
    logger.debug("write_results(%i, %s)", len(res_dict), filename)
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, **CSV_PARAMS)
        # pharmacological activity classes
        writer.writerow(("", "", *(col_classes[col] for col in cols)))
        # pharmacological activities
        writer.writerow(("", "", *cols))
        # write compounds, one by line
        for row in rows:
            values = [res_dict[row].get(col, -1) for col in cols]  # type:ignore
            writer.writerow((row_classes[row], row, *values))


def standardize(string: str) -> str:
    """Standardize strings"""
    return string.strip().lower()


# ( KEY ( terpenoid  OR  terpene ) AND KEY ( stimulant ) ) : 32
# ( KEY ( terpenoid) OR KEY (terpene )) AND KEY ( stimulant ) : 32
# ( KEY ( terpenoid) ) AND KEY ( stimulant ) : 8
# ( KEY ( terpene) ) AND KEY ( stimulant ) : 24
def build_search_query(keywords1, keywords2):
    """transform two keywords into a SCOPUS API query string. Splits keywords as OR subqueries if needed"""

    def slashes_to_or(string):
        return " OR ".join(string.split("/"))

    disjunct1 = slashes_to_or(keywords1)
    disjunct2 = slashes_to_or(keywords2)
    return {
        "query": f'( KEY ( {disjunct1} ) AND KEY ( {disjunct2} ) ) AND ( DOCTYPE( "ar" ) )',
        "count": 1,
    }


async def main_gather_all(pairs, mode):
    """OLD : Create tasks sequentially with throttling : one query every 1/MAX_REQ_BY_SEC second"""
    coros = []
    # pick the query mode
    task_factory = MODES.get(mode, MODES["fake"])
    # build all the tasks, each one sending an http network request over the network after some delay
    # delay is incremented by 1/MAX_REQ_BY_SEC
    async with aiohttp.ClientSession() as session:
        for i, (chemo, pharma) in enumerate(pairs):
            delay = i * 1 / DEFAULT_WORKERS
            task = asyncio.create_task(task_factory(session, chemo, pharma, delay=delay))
            coros.append(task)
        logger.info(
            "main: %i jobs created, estimated duration %f sec",
            len(pairs),
            len(pairs) / DEFAULT_WORKERS + BASE_RESPONSE_TIME,
        )
        res = await asyncio.gather(*coros)

    res_dict = defaultdict(dict)
    for (chemo, pharma, nb_results, _) in res:
        res_dict[chemo][pharma] = nb_results
    return res_dict


async def query_fake(_, keyword1, keyword2, *, delay=0):
    """Fake query tool without network, for test purpose"""
    results_nb = randint(1, 10000)
    await asyncio.sleep(delay)
    start_time = time.perf_counter()
    logger.debug("query_fake(%s, %s): launching at %s", keyword1, keyword2, start_time)
    logger.debug("           %s", build_search_query(keyword1, keyword2)["query"])

    await asyncio.sleep(randint(1, 1000) / 1000)
    elapsed = time.perf_counter() - start_time
    return (keyword1, keyword2, results_nb, elapsed)


async def query_httpbin(session, keyword1, keyword2, *, delay=0):
    """Fake query tool WITH network on httpbin, for test purpose"""
    # simule 1% d'erreur
    if randint(1, 100) <= ERROR_RATE:
        url = "http://httpbin.org/status/429"
    else:
        url = "http://httpbin.org/anything"
    data = {"answer": randint(1, 10000)}
    results_nb = -1
    await asyncio.sleep(delay)
    start_time = time.perf_counter()
    logger.debug("query_httpbin(%s, %s): launching at %s", keyword1, keyword2, start_time)
    try:
        async with session.get(url, params=build_search_query(keyword1, keyword2), data=data, ssl=SSL_CONTEXT) as resp:
            json = await resp.json()
            # args = json["args"]["query"]
            results_nb = int(json["form"]["answer"])
            logger.debug("query_httpbin(%s, %s): results_nb=%i", keyword1, keyword2, results_nb)
    except aiohttp.ClientError as err:  # aiohttp.ClientError
        logger.error(err)
        results_nb = -1
    finally:
        elapsed = time.perf_counter() - start_time
    return (keyword1, keyword2, results_nb, elapsed)


async def query_scopus(session, keyword1, keyword2, *, delay=0):
    """SCOPUS query tool: return the number of article papers having two sets of keywords. Delay is in sec"""
    scopus_url = "https://api.elsevier.com/content/search/scopus"
    results_nb = -1
    query = build_search_query(keyword1, keyword2)
    await asyncio.sleep(delay)
    start_time = time.perf_counter()
    logger.debug("query_scopus(%s, %s) @%s", keyword1, keyword2, start_time)
    try:
        async with session.get(scopus_url, params=query, headers=API_KEY, ssl=SSL_CONTEXT) as resp:
            logger.debug("X-RateLimit-Remaining=%s", resp.headers.get("X-RateLimit-Remaining", None))
            json = await resp.json()
            results_nb = int(json["search-results"]["opensearch:totalResults"], 10)
            logger.debug("query_scopus(%s, %s): results_nb=%i", keyword1, keyword2, results_nb)
    except aiohttp.ClientError as err:
        logger.error(err)
        results_nb = -1
    finally:
        elapsed = time.perf_counter() - start_time
    return (keyword1, keyword2, results_nb, elapsed)


# query modes : one fake, one fake over network, one true
MODES = {"scopus": query_scopus, "httpbin": query_httpbin, "fake": query_fake}
BASE_RESPONSE_TIME = 1.0


async def produce(queue, queries):
    """Produce jobs (queries) into the job queue"""
    for i, query in enumerate(queries):
        await queue.put(query)
        logger.debug("produce(): created job #%i query=%s", i, query)


async def consume(session, queue, res_dict, task_factory, delay=1, name=None):
    """A (parallel) consumer that send a query to scopus and then add result to a dict"""
    jobs_done = 0
    try:
        while True:
            (kw1, kw2) = await queue.get()
            (chemo, pharma, nb_results, duration) = await task_factory(session, kw1, kw2, delay=0)
            logger.info("consume(%s): got %s from job %s after %f", name, nb_results, (chemo, pharma), duration)
            await asyncio.sleep(max(delay - duration, 0))
            res_dict[chemo][pharma] = nb_results
            queue.task_done()
            jobs_done += 1
    except asyncio.CancelledError:
        logger.debug("task %s received cancel, done %i jobs", name, jobs_done)
    return jobs_done


async def main_queue(queries, parallel, mode, delay):
    """Create tasks in a queue which is emptied in parallele ensuring at most MAX_REQ_BY_SEC requests per second"""
    jobs = asyncio.Queue()
    res_dict = defaultdict(dict)
    task_factory = MODES.get(mode, MODES["fake"])

    # ONE producer task to fill the queue
    producer_task = asyncio.create_task(produce(jobs, queries), name="producer")
    # MAX_REQ_BY_SEC consummers that run in parallel and that can fire at most
    # one request per second
    consumer_tasks = []
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        consumer_tasks = [
            asyncio.create_task(
                consume(session, jobs, res_dict, task_factory, delay=delay, name=i), name=f"consumer-{i}"
            )
            for i in range(1, parallel + 1)
        ]

        logger.info("Tasks created")
        await asyncio.gather(producer_task)

        # on attend que tout soit traité, après que tout soit généré
        await jobs.join()
        logger.info("All jobs dones")
        # stop all consumer stuck waiting job from the queue
        for consumer in consumer_tasks:
            consumer.cancel()
        logger.debug("Consumers cancellation ordered")
        nb_jobs_done = await asyncio.gather(*consumer_tasks)
        logger.info("All consumers cancelled, jobs done : %s", nb_jobs_done)

    # pending = asyncio.all_tasks()
    # logger.debug(pending)

    return res_dict


def sorted_keys(classes, base_only=True):
    """Ensure that parents classes are first in order, and that children are classes-wise ordered alphabeticaly"""
    return sorted(
        (com for com, cls in classes.items() if not base_only or cls == BASE_CLASS), key=lambda x: (classes[x], x)
    )


# %%

# download_all(mode=args.mode, parallel=args.parallel, , samples=args.samples, all=args.all, write=args.write)
def download_all(
    mode=DEFAULT_MODE,
    parallel=DEFAULT_WORKERS,
    samples=None,
    all_classes=False,
    no_write=False,
    delay=DEFAULT_DELAY_PER_WORKER,
):
    """Launch the batch of downloads"""
    compounds = get_classes(COMPOUNDS)
    pharmaco = get_classes(PHARMACOLOGY)
    compounds_keywords = sorted_keys(compounds, base_only=not all_classes)
    pharmaco_keywords = sorted_keys(pharmaco, base_only=not all_classes)
    if samples is None:
        queries = list(product(compounds_keywords, pharmaco_keywords))
    else:
        queries = sample(list(product(compounds_keywords, pharmaco_keywords)), samples)

    logger.info("%i compounds X %i pharmacology = %i", len(compounds_keywords), len(pharmaco_keywords), len(queries))

    main_start_time = time.perf_counter()
    logger.info("START")

    # correction bug scopus
    # results = asyncio.run(main_queue(queries, mode=mode))
    loop = asyncio.get_event_loop()
    main_task = loop.create_task(
        main_queue(queries=queries, parallel=parallel, mode=mode, delay=delay), name="main-queue"
    )
    results = loop.run_until_complete(main_task)

    total_time = time.perf_counter() - main_start_time
    logger.info("DONE in %fs", total_time)

    if not no_write:
        now = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
        output_filename = OUTPUT_DIR / f"activity_{now}.csv"
        write_results(results, compounds_keywords, pharmaco_keywords, compounds, pharmaco, output_filename)
        logger.info("WRITTEN %s", output_filename)


def load_results(filename: Path, *, chemo_cls_nb, pharm_cls_nb, pharm_nb):
    """Loads chemical compounds / pharmacological activity matrice from SCOPUS"""
    logger.debug("load_results(%s)", filename)
    usecols = range(2, pharm_nb + 2)
    full_matrix = np.loadtxt(filename, dtype=np.int32, delimiter=";", skiprows=2, encoding="utf-8", usecols=usecols)

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


def get_parser():
    """argparse configuration"""
    arg_parser = argparse.ArgumentParser(description="Scopus downloader")
    arg_parser.add_argument(
        "--verbose", "-v", action="store_true", default=False, help="verbosity level set to DEBUG, default is INFO"
    )
    arg_parser.add_argument(
        "--mode",
        "-m",
        action="store",
        default=DEFAULT_MODE,
        help="download mode: 'mock', 'httpbin' or 'scopus'. Default is {DEFAULT_MODE}",
    )
    arg_parser.add_argument(
        "--parallel",
        "-p",
        type=int,
        action="store",
        default=DEFAULT_WORKERS,
        help=f"number of parallel consumers/workers, default {DEFAULT_WORKERS}",
    )
    arg_parser.add_argument(
        "--delay",
        "-d",
        type=float,
        action="store",
        default=DEFAULT_DELAY_PER_WORKER,
        help=f"delay between consecutive queries from a worker, default {DEFAULT_DELAY_PER_WORKER}",
    )
    arg_parser.add_argument(
        "--samples",
        "-s",
        type=int,
        action="store",
        default=DEFAULT_SAMPLES,
        help="maximum number of queries (random sample), default None (all)",
    )
    arg_parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        default=False,
        help="download both parent categories and children",
    )
    arg_parser.add_argument(
        "--no-write",
        "-w",
        action="store_true",
        default=False,
        help="do not write results to csv file (default False)",
    )
    return arg_parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # https://docs.python.org/3/library/logging.html#levels
    if args.verbose:
        LEVEL = logging.DEBUG
    else:
        LEVEL = logging.INFO
    logger.setLevel(LEVEL)

    download_all(
        mode=args.mode,
        parallel=args.parallel,
        samples=args.samples,
        all_classes=args.all,
        no_write=args.no_write,
        delay=args.delay,
    )
    # pass

# if __name__ == "__main__":
#     asyncio.run(run_async_query("phenolic compound", "chronic disease", 0))
# 38 articles et 69 au total

# py .\downloader.py --mode httpbin --no-write -p 10
# py .\downloader.py --mode httpbin --no-write --delay 0.0 --parallel 55 --samples 100 --all

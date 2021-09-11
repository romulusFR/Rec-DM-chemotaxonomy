"""test de queries async à scopus via aoihttp"""
# pylint: disable=unused-import
# %%

import asyncio
import csv
import datetime
import logging
from collections import defaultdict
from itertools import product
from pathlib import Path
from pprint import pprint
from random import randint, sample
from typing import Tuple

import aiohttp
import pandas as pd

logging.basicConfig()
logger = logging.getLogger("scopus_api")
logger.setLevel(logging.INFO)

# https://www.twilio.com/blog/asynchronous-http-requests-in-python-with-aiohttp
# https://stackoverflow.com/questions/48682147/aiohttp-rate-limiting-parallel-requests
# https://docs.aiohttp.org/en/stable/client.html
# https://docs.aiohttp.org/en/stable/client_advanced.html


# SAMPLES / TESTS
CHEMISTRY = ["alkaloid", "polyphenol", "coumarin"]
ACTIVITIES = ["antiinflammatory", "anticoagulant", "cancer"]
RESULTS = {"acridine": {"anticancer": "2790"}, "pyridine": {"toxicant": "1904"}, "tetracycline": {"repulsive": "4598"}}
QUERIES = list(product(CHEMISTRY, ACTIVITIES))
NB_MAX_TEST_QUERIES = 5

# DATASETS
COMPOUNDS = Path("data") / "compounds.csv"
PHARMACOLOGY = Path("data") / "pharmacology.csv"
BASE_CLASS = "*"

# OUTPUT
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PARAMS = {"delimiter": ";", "quotechar": '"'}

# API Scopus
MAX_REQ_BY_SEC = 8
API_KEY = {"X-ELS-APIKey": "7047b3a8cf46d922d5d5ca71ff531b7d"}
X_RATE_HEADERS = ["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"]


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


async def query_fake(keyword1, keyword2, /, delay=1):
    """Fake query tool without network, for test purpose"""
    results_nb = randint(1, 10000)
    await asyncio.sleep(delay)
    start_time = datetime.datetime.now()
    logger.debug("query_fake(%s, %s): launching at %s", keyword1, keyword2, start_time)
    logger.debug("           %s", build_search_query(keyword1, keyword2)["query"])

    await asyncio.sleep(randint(1, 1000) / 1000)
    elapsed = datetime.datetime.now() - start_time
    return (keyword1, keyword2, results_nb, elapsed.seconds + elapsed.microseconds / 10 ** 6)


async def query_httpbin(keyword1, keyword2, /, delay=1):
    """Fake query tool WITH network on httpbin, for test purpose"""
    url = "http://httpbin.org/anything"
    data = {"answer": randint(1, 10000)}
    results_nb = -1
    await asyncio.sleep(delay)
    start_time = datetime.datetime.now()
    logger.debug("query_httpbin(%s, %s): launching at %s", keyword1, keyword2, start_time)
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, params=build_search_query(keyword1, keyword2), data=data) as resp:
                json = await resp.json()
                args = json["args"]["query"]
                results_nb = json["form"]["answer"]
                msg = f"run_async_query_test: query={args} results_nb={results_nb}"
                logger.debug(msg)
        except Exception as err:
            logger.error(err)
            results_nb = -1
        finally:
            elapsed = datetime.datetime.now() - start_time
        return (keyword1, keyword2, results_nb, elapsed.seconds + elapsed.microseconds / 10 ** 6)


async def query_scopus(keyword1, keyword2, /, delay=1):
    """SCOPUS query tool: return the number of article papers having two sets of keywords. Delay is in sec"""
    scopus_url = "https://api.elsevier.com/content/search/scopus"
    results_nb = -1
    await asyncio.sleep(delay)
    start_time = datetime.datetime.now()
    logger.debug("query_scopus(%s, %s) @%s", keyword1, keyword2, start_time)
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(scopus_url, params=build_search_query(keyword1, keyword2), headers=API_KEY) as resp:
                logger.debug("X-RateLimit-Remaining=%s", resp.headers.get("X-RateLimit-Remaining", None))
                json = await resp.json()
                results_nb = int(json["search-results"]["opensearch:totalResults"], 10)
                logger.debug(
                    "run_async_query(%s, %s) = %i @%s", keyword1, keyword2, results_nb, datetime.datetime.now()
                )
        except Exception as err:
            logger.error(err)
            results_nb = -1
        finally:
            elapsed = datetime.datetime.now() - start_time
        return (keyword1, keyword2, results_nb, elapsed.seconds + elapsed.microseconds / 10 ** 6)


# query modes : one fake, one fake over network, one true
MODES = {"scopus": query_scopus, "httpbin": query_httpbin, "fake": query_fake}
BASE_RESPONSE_TIME = 1.0


async def main_gather_all(pairs, mode):
    """Create tasks sequentially with throttling : one query every 1/MAX_REQ_BY_SEC second"""
    coros = []
    # pick the query mode
    task_factory = MODES.get(mode, MODES["fake"])
    # build all the tasks, each one sending an http network request over the network after some delay
    # delay is incremented by 1/MAX_REQ_BY_SEC
    for i, (chemo, pharma) in enumerate(pairs):
        delay = i * 1 / MAX_REQ_BY_SEC
        task = asyncio.create_task(task_factory(chemo, pharma, delay=delay))
        coros.append(task)
    logger.info(
        "main: %i jobs created, estimated duration %f sec", len(pairs), len(pairs) / MAX_REQ_BY_SEC + BASE_RESPONSE_TIME
    )

    res = await asyncio.gather(*coros)
    res_dict = defaultdict(dict)
    for (chemo, pharma, nb_results, _) in res:
        res_dict[chemo][pharma] = nb_results
    return res_dict


async def produce(queue, queries):
    """Produce jobs (queries) into the job queue"""
    for i, query in enumerate(queries):
        await queue.put(query)
        logger.debug("produce(): created job #%i query=%s", i, query)


async def consume(queue, res_dict, task_factory, delay=1, name=None):
    """A (parallel) consumer that send a query to scopus and then add result to a dict"""
    while True:
        (chemo, pharma) = await queue.get()

        (chemo, pharma, nb_results, duration) = await task_factory(chemo, pharma, delay=delay)
        logger.info("consume(%s): got %s from job %s after %f", name, nb_results, (chemo, pharma), duration)
        await asyncio.sleep(max(delay - duration, 0))
        res_dict[chemo][pharma] = nb_results
        queue.task_done()


async def main_queue(pairs, mode):
    """Create tasks in a queue which is emptied in parallele ensuring at most MAX_REQ_BY_SEC requests per second"""
    jobs = asyncio.Queue()
    res_dict = defaultdict(dict)
    task_factory = MODES.get(mode, MODES["fake"])

    # ONE producer task to fill the queue
    producer_task = asyncio.create_task(produce(jobs, pairs), name="producer")
    # MAX_REQ_BY_SEC consummers that run in parallel and that can fire at most
    # one request per second
    consumer_tasks = [
        asyncio.create_task(consume(jobs, res_dict, task_factory, delay=1, name=i), name=f"consumer-{i}")
        for i in range(MAX_REQ_BY_SEC)
    ]

    logger.info("Tasks created")
    await asyncio.gather(producer_task)

    # on attend que tout soit traité, après que tout soit généré
    await jobs.join()
    logger.info("All jobs dones")
    # stop all consumer stuck waiting job from the queue
    for consumer in consumer_tasks:
        consumer.cancel()
    return res_dict


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


def sorted_keys(classes, base_only=True):
    """Ensure that parents classes are first in order, and that children are classes-wise ordered alphabeticaly"""
    return sorted(
        (com for com, cls in classes.items() if not base_only or cls == BASE_CLASS), key=lambda x: (classes[x], x)
    )


# %%


def download_all(mode="mock", base_only=True):
    """Launch the batch of downloads"""
    compounds = get_classes(COMPOUNDS)
    pharmaco = get_classes(PHARMACOLOGY)
    compounds_keywords = sorted_keys(compounds, base_only=base_only)
    pharmaco_keywords = sorted_keys(pharmaco, base_only=base_only)
    queries = list(product(compounds_keywords, pharmaco_keywords))
    logger.info("%i compounds X %i pharmacology = %i", len(compounds_keywords), len(pharmaco_keywords), len(queries))

    main_start_time = datetime.datetime.now()
    logger.info("START")
    results = asyncio.run(main_queue(queries, mode=mode))
    total_time = datetime.datetime.now() - main_start_time
    logger.info("DONE in %fs", total_time.seconds + total_time.microseconds / 10 ** 6)

    output_filename = OUTPUT_DIR / f"activity_{datetime.datetime.now()}.csv"
    # output_filename = OUTPUT_DIR / f"activity_{'test'}.csv"
    write_results(results, compounds_keywords, pharmaco_keywords, compounds, pharmaco, output_filename)
    logger.info("WRITTEN %s", output_filename)


if __name__ == "__main__":
    download_all(mode="mock", base_only=True)
    # pass

# if __name__ == "__main__":
#     asyncio.run(run_async_query("phenolic compound", "chronic disease", 0))
# 38 articles et 69 au total

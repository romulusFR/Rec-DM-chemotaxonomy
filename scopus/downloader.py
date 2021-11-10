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

from loader import Dataset, load_chemo_activities, write_chemo_activities


logging.basicConfig()
logger = logging.getLogger("scopus_api")
logger.setLevel(logging.DEBUG)

# https://www.twilio.com/blog/asynchronous-http-requests-in-python-with-aiohttp
# https://stackoverflow.com/questions/48682147/aiohttp-rate-limiting-parallel-requests
# https://docs.aiohttp.org/en/stable/client.html
# https://docs.aiohttp.org/en/stable/client_advanced.html


# OUTPUT
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# API Scopus
API_KEY = {"X-ELS-APIKey": "7047b3a8cf46d922d5d5ca71ff531b7d"}
X_RATE_HEADERS = ["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"]
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

# DEFAULT VALUES
DEFAULT_WORKERS = 8
DEFAULT_DELAY_PER_WORKER = 1.0
DEFAULT_SAMPLES = None
DEFAULT_MODE = "mock"


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


async def query_httpbin(session, keyword1, keyword2, *, delay=0, error_rate=1):
    """Fake query tool WITH network on httpbin, for test purpose"""
    # simule 1% d'erreur
    if randint(1, 100) <= error_rate:
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


# %%

# download_all(mode=args.mode, parallel=args.parallel, , samples=args.samples, all=args.all, write=args.write)
def download_all(
    dataset: Dataset,
    mode=DEFAULT_MODE,
    parallel=DEFAULT_WORKERS,
    samples=None,
    delay=DEFAULT_DELAY_PER_WORKER,
    no_write=False,
):
    """Launch the batch of downloads"""
    compounds = dataset.compounds.keys()
    activities = dataset.activities.keys()
    if samples is None:
        queries = list(product(compounds, activities))
    else:
        queries = sample(list(product(compounds, activities)), samples)

    logger.info("%i compounds X %i pharmacology = %i", len(compounds), len(activities), len(queries))

    main_start_time = time.perf_counter()
    logger.info("START")

    # correction bug scopus
    # results = asyncio.run(main_queue(queries, mode=mode))
    loop = asyncio.get_event_loop()
    main_task = loop.create_task(
        main_queue(queries=queries, parallel=parallel, mode=mode, delay=delay), name="main-queue"
    )
    results = loop.run_until_complete(main_task)
    dataset.data = results

    total_time = time.perf_counter() - main_start_time
    logger.info("DONE in %fs", total_time)

    if not no_write:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_filename = OUTPUT_DIR / f"activities_{now}.csv"
        write_chemo_activities(output_filename, dataset)
        logger.info("WRITTEN %s", output_filename)


def get_parser():
    """argparse configuration"""
    arg_parser = argparse.ArgumentParser(description="Scopus downloader")
    arg_parser.add_argument(
        "filename",
        help="file to read chemo activities from",
    )
    arg_parser.add_argument(
        "--verbose", "-v", action="store_true", default=False, help="verbosity level set to DEBUG (default is INFO)"
    )
    arg_parser.add_argument(
        "--mode",
        "-m",
        action="store",
        default=DEFAULT_MODE,
        help=f"download mode: 'mock', 'httpbin' or 'scopus' (default '{DEFAULT_MODE}')",
    )
    arg_parser.add_argument(
        "--parallel",
        "-p",
        type=int,
        action="store",
        default=DEFAULT_WORKERS,
        help=f"number of parallel consumers/workers (default {DEFAULT_WORKERS})",
    )
    arg_parser.add_argument(
        "--delay",
        "-d",
        type=float,
        action="store",
        default=DEFAULT_DELAY_PER_WORKER,
        help=f"delay between consecutive queries from a worker (default {DEFAULT_DELAY_PER_WORKER})",
    )
    arg_parser.add_argument(
        "--samples",
        "-s",
        type=int,
        action="store",
        default=DEFAULT_SAMPLES,
        help="maximum number of queries (random samples) (default all pairs)",
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

    dataset = load_chemo_activities(args.filename)
    download_all(
        dataset,
        mode=args.mode,
        parallel=args.parallel,
        samples=args.samples,
        no_write=args.no_write,
        delay=args.delay,
    )
    # pass

# if __name__ == "__main__":
#     asyncio.run(run_async_query("phenolic compound", "chronic disease", 0))
# 38 articles et 69 au total

# py .\downloader.py --mode httpbin --no-write -p 10
# py .\downloader.py --mode httpbin --no-write --delay 0.0 --parallel 55 --samples 100 --all

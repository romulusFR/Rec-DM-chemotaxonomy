"""test de queries async à scopus via aoihttp"""
# pylint: disable=unused-import
# %%

import argparse
import asyncio
import csv
import logging
import ssl
import time
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
from itertools import product, chain
from pathlib import Path
from pprint import pprint, pformat
from random import randint, sample
from typing import Tuple, List
from os import environ

from dotenv import load_dotenv
import certifi
import aiohttp

# https://www.twilio.com/blog/asynchronous-http-requests-in-python-with-aiohttp
# https://stackoverflow.com/questions/48682147/aiohttp-rate-limiting-parallel-requests
# https://docs.aiohttp.org/en/stable/client.html
# https://docs.aiohttp.org/en/stable/client_advanced.html


from loader import Dataset, load_chemo_activities, write_chemo_activities


logging.basicConfig()
logger = logging.getLogger("scopus_api")
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

# DEFAULT VALUES
DEFAULT_WORKERS = 8
DEFAULT_DELAY_PER_WORKER = 1.0
DEFAULT_SAMPLES = None
DEFAULT_MODE = "fake"


# ( KEY ( terpenoid  OR  terpene ) AND KEY ( stimulant ) ) : 32
# ( KEY ( terpenoid) OR KEY (terpene )) AND KEY ( stimulant ) : 32
# ( KEY ( terpenoid) ) AND KEY ( stimulant ) : 8
# ( KEY ( terpene) ) AND KEY ( stimulant ) : 24
def build_dnf_query(*disjuncts):
    """transform lists keywords into a SCOPUS API query string.
       Splits keywords as OR subqueries if needed.
       Variadic parameter : the conjuncts. Each parameter is a list of disjuncts.

       Example :

       build_dnf_query(["pyridine", "oxazole", "terpenoid / terpene"], ["toxicity", "cancer/death"])

       Creates the following query string

       {'query': '( KEY ( pyridine OR oxazole OR terpenoid OR terpene ) AND KEY ( toxicity OR cancer OR death ) ) AND ( DOCTYPE( "ar" ) )',
    'count': 1}

    """

    clauses = [
        " OR ".join(keyword.strip() for coumpound_keyword in disjunct for keyword in coumpound_keyword.split("/"))
        for disjunct in disjuncts
    ]
    # KEY ( {disjunct1} ) AND KEY ( {disjunct2} )
    clausal_form = " AND ".join(f"KEY ( {clause} )" for clause in clauses if len(clause))

    return {
        "query": f'( {clausal_form} ) AND ( DOCTYPE( "ar" ) )',
        "count": 1,
    }


async def query_fake(_, kw_clauses, *, delay=0):
    """Fake query tool without network, for test purpose"""
    results_nb = randint(1, 10000)
    await asyncio.sleep(delay)
    start_time = time.perf_counter()
    logger.debug("query_fake(%s): launching at %s", kw_clauses, start_time)
    logger.debug("           %s", build_dnf_query(*kw_clauses)["query"])

    await asyncio.sleep(randint(1, 1000) / 1000)
    elapsed = time.perf_counter() - start_time
    return (kw_clauses, results_nb, elapsed)


async def query_httpbin(session, kw_clauses, *, delay=0, error_rate=1):
    """Fake query tool WITH network on httpbin, for test purpose"""
    # simule 1% d'erreur
    if randint(1, 100) <= error_rate:
        url = "http://httpbin.org/status/429"
    else:
        url = "http://httpbin.org/anything"
    data = {"answer": randint(1, 10000)}
    results_nb = -1
    query = build_dnf_query(*kw_clauses)
    await asyncio.sleep(delay)
    start_time = time.perf_counter()
    logger.debug("query_httpbin(%s): launching at %s", kw_clauses, start_time)
    logger.debug("           %s", query)

    try:
        async with session.get(url, params=query, data=data, ssl=SSL_CONTEXT) as resp:
            json = await resp.json()
            # args = json["args"]["query"]
            results_nb = int(json["form"]["answer"])
            logger.debug("query_httpbin(%s): results_nb=%i", kw_clauses, results_nb)
    except aiohttp.ClientError as err:  # aiohttp.ClientError
        logger.error(err)
        results_nb = -1
    finally:
        elapsed = time.perf_counter() - start_time
    return (kw_clauses, results_nb, elapsed)


async def query_scopus(session, kw_clauses, *, delay=0):
    """SCOPUS query tool: return the number of article papers having two sets of keywords. Delay is in sec"""
    scopus_url = "https://api.elsevier.com/content/search/scopus"
    results_nb = -1
    query = build_dnf_query(*kw_clauses)
    await asyncio.sleep(delay)
    start_time = time.perf_counter()
    logger.debug("query_scopus(%s) @%s", kw_clauses, start_time)
    logger.debug("           %s", query)
    try:
        async with session.get(scopus_url, params=query, headers=API_KEY, ssl=SSL_CONTEXT) as resp:
            logger.debug("X-RateLimit-Remaining=%s", resp.headers.get("X-RateLimit-Remaining", None))
            json = await resp.json()
            results_nb = int(json["search-results"]["opensearch:totalResults"], 10)
            logger.debug("query_scopus(%s): results_nb=%i", kw_clauses, results_nb)
    except aiohttp.ClientError as err:
        logger.error(err)
        results_nb = -1
    finally:
        elapsed = time.perf_counter() - start_time
    return (kw_clauses, results_nb, elapsed)


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
            keywords = await queue.get()
            (keywords, nb_results, duration) = await task_factory(session, keywords, delay=0)
            logger.info("consume(id=%s): got %s from job %s after %f", name, nb_results, keywords, duration)
            await asyncio.sleep(max(delay - duration, 0))
            if len(keywords) == 2:
                [chemo, pharma] = keywords
                res_dict[chemo][pharma] = nb_results
                logger.debug("consume(id=%s): (%s,%s)=%i added)", name, chemo, pharma, nb_results)
            elif len(keywords) == 1:
                [item] = keywords
                res_dict[item] = nb_results
                logger.debug("consume(id=%s): (%s,)=%i added)", name, item, nb_results)
            else:
                logger.warning("consume(id=%s): unable to deal with %i keywords %s", name, len(keywords), keywords)
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
    logger.info("download mode/function is '%s'", task_factory.__name__)

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

        logger.info("%i tasks created", len(consumer_tasks))
        await asyncio.gather(producer_task)

        # on attend que tout soit traité, après que tout soit généré
        await jobs.join()
        # stop all consumer stuck waiting job from the queue
        for consumer in consumer_tasks:
            consumer.cancel()
        logger.debug("consumers cancellation ordered")
        nb_jobs_done = await asyncio.gather(*consumer_tasks)
        logger.info("all consumers cancelled, jobs done : %s", nb_jobs_done)

    # pending = asyncio.all_tasks()
    # logger.debug(pending)

    return res_dict

def splits_slashes(coumpound_keyword) : 
    """Divides and normalizes a string along slashes (/) to a tuples of strings
    
    examples

    splits_slashes("triterpene") == ('triterpene',)
    splits_slashes("terpenoid / terpene") == ('terpenoid', 'terpene')
    """
    return tuple(keyword.strip() for keyword in coumpound_keyword.split("/"))

@dataclass
class ScopusQuery(repr=False):
    any_compounds : Tuple[str]
    but_not_compounds : Tuple[str]
    any_activities : Tuple[str]
    but_not_activities : Tuple[str]
    
    def to_scopus(self) -> str:
    #     clauses = [
    #     " OR ".join(keyword.strip() for coumpound_keyword in disjunct for keyword in coumpound_keyword.split("/"))
    #     for disjunct in disjuncts
    # ]
    # # KEY ( {disjunct1} ) AND KEY ( {disjunct2} )
    # clausal_form = " AND ".join(f"KEY ( {clause} )" for clause in clauses if len(clause))

        return {
            "query": f'DOCTYPE( "ar" ) AND ( {clausal_form} ) ',
            "count": 1,
        }

def generate_all_dnf(data: Dataset) :

    compounds: List[str] = list(data.compounds.keys())
    activities: List[str] = list(data.activities.keys())
    any_compound = chain(splits_slashes(compound) for compound in compounds)
    any_activity = chain(splits_slashes(activity) for activity in activities)

    queries = []
    # grand total
    queries.push( ()  )
    # # marginal sums : rows
    
    # queries += [((splits_slashes(compound,), tuple(any_activity)), ()) for compound in compounds]
    # # marginal sums : cols
    # queries += [((tuple(compounds), (activity,)), ()) for activity in activities]

    # queries = [[[compound], [activity]] for compound in compounds for activity in activities]
    # queries += [[compounds], [activities]]
    # queries += [[compounds], [activities]]
    # queries += [[compounds, activities]]
    logger.debug(f"generate_all_dnf({compounds}, {activities}) =\n{pformat(queries)}")
    return queries


# download_all(mode=args.mode, parallel=args.parallel, , samples=args.samples, all=args.all, write=args.write)
def download_all(
    dataset: Dataset,
    mode=DEFAULT_MODE,
    parallel=DEFAULT_WORKERS,
    samples=None,
    delay=DEFAULT_DELAY_PER_WORKER,
    write=False,
):
    """Launch the batch of downloads"""

    # here, constructs all pairs {(c, a) | c in compounds, a in activity}
    # list(product(compounds, activities))
    queries = generate_all_dnf(dataset)
    if samples is not None:
        queries = sample(queries, samples)  # [(c,) for c in compounds]

    logger.info("%i compounds X %i pharmacology = %i", len(dataset.compounds), len(dataset.activities), len(queries))

    main_start_time = time.perf_counter()
    logger.info("starting filling queue with jobs")

    # correction bug scopus
    # results = asyncio.run(main_queue(queries, mode=mode))
    loop = asyncio.get_event_loop()
    main_task = loop.create_task(
        main_queue(queries=queries, parallel=parallel, mode=mode, delay=delay), name="main-queue"
    )
    logger.info("launching jobs")
    results = loop.run_until_complete(main_task)
    dataset.data = results

    total_time = time.perf_counter() - main_start_time
    logger.info("all jobs done in %fs", total_time)

    if write:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_filename = OUTPUT_DIR / f"activities_{now}.csv"
        write_chemo_activities(output_filename, dataset)
        logger.info("WRITTEN %s", output_filename)


# %%
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
        help=f"download mode: 'fake', 'httpbin' or 'scopus' (default '{DEFAULT_MODE}')",
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
        "--write",
        "-w",
        action="store_true",
        default=False,
        help="do write results to csv file (default False)",
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

    dataset_content = load_chemo_activities(args.filename)
    download_all(
        dataset=dataset_content,
        mode=args.mode,
        parallel=args.parallel,
        samples=args.samples,
        write=args.write,
        delay=args.delay,
    )
    # pass

# if __name__ == "__main__":
#     asyncio.run(run_async_query("phenolic compound", "chronic disease", 0))
# 38 articles et 69 au total

# py .\downloader.py --mode httpbin --no-write -p 10
# py .\downloader.py --mode httpbin --no-write --delay 0.0 --parallel 55 --samples 100 --all

if False:
    dataset_content = load_chemo_activities("data/tests.csv")
    generate_all_dnf(dataset_content)

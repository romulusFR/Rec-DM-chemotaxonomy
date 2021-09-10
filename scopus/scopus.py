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
logger.setLevel(logging.DEBUG)

# https://www.twilio.com/blog/asynchronous-http-requests-in-python-with-aiohttp
# https://stackoverflow.com/questions/48682147/aiohttp-rate-limiting-parallel-requests
# https://docs.aiohttp.org/en/stable/client.html
# https://docs.aiohttp.org/en/stable/client_advanced.html


MAX_REQ_BY_SEC = 5
CHEMISTRY = ["alkaloid", "polyphenol", "coumarin"]
ACTIVITIES = ["antiinflammatory", "anticoagulant", "cancer"]

QUERIES = list(product(CHEMISTRY, ACTIVITIES))
NB_MAX_TEST_QUERIES = 5
CSV_PARAMS = {"delimiter": ";", "quotechar": '"'}

API_KEY = {"X-ELS-APIKey": "7047b3a8cf46d922d5d5ca71ff531b7d"}
search_and = lambda s1, s2: {
    "query": f'( KEY ( {s1} ) AND KEY ( {s2} ) ) AND ( DOCTYPE( "ar" ) )',
    "count": 1,
}
x_rate_limit_headers = ["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"]


async def run_async_query_test(keyword1, keyword2, /, delay=0):
    """launcher demo"""
    url = "http://httpbin.org/anything"
    data = {"answer": randint(1, 10000)}
    results_nb = -1
    await asyncio.sleep(delay)
    async with aiohttp.ClientSession() as session:
        start_time = datetime.datetime.now()
        logger.debug("run_async_query_test(%s, %s): launching at %s", keyword1, keyword2, start_time)
        try:
            async with session.get(url, params=search_and(keyword1, keyword2), data=data) as resp:
                json = await resp.json()
                args = json["args"]["query"]
                results_nb = json["form"]["answer"]
                msg = f"run_async_query_test: query={args} results_nb={results_nb}"
                logger.debug(msg)
        except aiohttp.ClientError as err:
            logger.error(err)
            results_nb = -1
        finally:
            elapsed = datetime.datetime.now() - start_time
        return (keyword1, keyword2, results_nb, elapsed.seconds + elapsed.microseconds / 10 ** 6)


async def run_async_query(keyword1, keyword2, /, delay):
    """launcher"""
    scopus_url = "https://api.elsevier.com/content/search/scopus"
    results_nb = -1
    await asyncio.sleep(delay)
    start_time = datetime.datetime.now()
    logger.debug("run_async_query(%s, %s) @%s", keyword1, keyword2, start_time)
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(scopus_url, params=search_and(keyword1, keyword2), headers=API_KEY) as resp:
                logger.debug("X-RateLimit-Remaining=%s", resp.headers.get("X-RateLimit-Remaining", None))
                json = await resp.json()
                results_nb = int(json["search-results"]["opensearch:totalResults"], 10)
                logger.debug(
                    "run_async_query(%s, %s) = %i @%s", keyword1, keyword2, results_nb, datetime.datetime.now()
                )
        except aiohttp.ClientError as err:
            logger.error(err)
            results_nb = -1
        finally:
            elapsed = datetime.datetime.now() - start_time
        return (keyword1, keyword2, results_nb, elapsed.seconds + elapsed.microseconds / 10 ** 6)


async def main(pairs, debug=True):
    """Create tasks sequentially with throttling"""
    coros = []
    task_factory = run_async_query_test if debug else run_async_query
    for i, (chemo, pharma) in enumerate(pairs):
        delay = i * 1 / MAX_REQ_BY_SEC
        task = asyncio.create_task(task_factory(chemo, pharma, delay=delay))
        coros.append(task)
    logger.info("main: %i jobs created", len(pairs))

    res = await asyncio.gather(*coros)
    res_dict = defaultdict(dict)
    for (chemo, pharma, nb, _) in res:
        res_dict[chemo][pharma] = nb
    return res_dict
    # return {(chemo, pharma): (nb, duration) for (chemo, pharma, nb, duration) in res}


def clean_word(string: str) -> str:
    """standardize"""
    return string.strip().lower()


def get_classes(filename):
    """Loads classes from a csv file with format (parent, child)"""
    logger.debug("get_classes(%s)", filename)
    classes = {}
    with open(filename, encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, **CSV_PARAMS)
        for (parent, child) in reader:
            classes[clean_word(child)] = clean_word(parent)
    return classes


def write_results(res_dict: dict[Tuple[str, str], Tuple[int, float]], rows, cols, filename):
    """Store result dict"""
    logger.debug("write_results(%i, %s)", len(results), filename)
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, **CSV_PARAMS)
        writer.writerow(("/", *cols))
        for row in rows:
            values = [res_dict[row].get(col, -1) for col in cols]  # type:ignore
            writer.writerow((row, *values))
    logger.debug("%s written", filename)


# RESULTS = {
#     ("isoflavanoids", "antibacterial"): ("26", 1.010633),
#     ("spermidine", "toxicity"): ("5503", 0.541761),
#     ("tropane", "hppd inhibitor"): ("2890", 0.543553),
# }

RESULTS = {"acridine": {"anticancer": "2790"}, "pyridine": {"toxicant": "1904"}, "tetracycline": {"repulsive": "4598"}}

# df = pd.DataFrame.from_dict(RESULTS, orient="index", columns=["articles", "query"])
# %%

if __name__ == "__main__":
    compounds = get_classes("compounds.csv")
    pharmaco = get_classes("pharmacology.csv")
    compounds_classes = {v for v in compounds.values()}
    pharmaco_classes = {v for v in pharmaco.values()}
    # queries = list(product(compounds, pharmaco))
    queries = list(product(compounds_classes, pharmaco_classes))

    main_start_time = datetime.datetime.now()
    logger.info("START")
    # results = asyncio.run(main(sample(queries, 1), debug=False))
    results = asyncio.run(main(queries), debug=False)
    total_time = datetime.datetime.now() - main_start_time
    logger.info("DONE in %fs", total_time.seconds + total_time.microseconds / 10 ** 6)

    # pprint(results)
    # write_results(results, compounds.keys(), pharmaco.keys(), "results.csv")
    Path("results").mkdir(parents=True, exist_ok=True)
    write_results(results, compounds_classes, pharmaco_classes, f"results/activity_{datetime.datetime.now()}.csv")


# if __name__ == "__main__":
#     asyncio.run(run_async_query("phenolic compound", "chronic disease", 0))

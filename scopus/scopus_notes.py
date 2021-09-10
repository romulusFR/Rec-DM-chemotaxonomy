"""test de queries async à scopus via aoihttp"""

# %%

import logging
import datetime
import asyncio
from pprint import pprint
from itertools import product
import aiohttp
from random import randint, sample

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


API_KEY = {"X-ELS-APIKey": "7047b3a8cf46d922d5d5ca71ff531b7d"}
search_and = lambda s1, s2: {"query": f"KEY({s1}) AND KEY({s2})", "count": 1}
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
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(scopus_url, params=search_and(keyword1, keyword2), headers=API_KEY) as resp:
                logger.debug("X-RateLimit-Remaining=%s", resp.headers.get("X-RateLimit-Remaining", None))
                json = await resp.json()
                results_nb = json["search-results"]["opensearch:totalResults"]
                msg = f"run_async_query({keyword1}, {keyword2})={results_nb}"
                logger.debug(msg)
        except aiohttp.ClientError as err:
            logger.error(err)
            results_nb = -1
        finally:
            elapsed = datetime.datetime.now() - start_time
        return (keyword1, keyword2, results_nb, elapsed.seconds + elapsed.microseconds / 10 ** 6)


async def main_wait_too_much(queries):
    """Create tasks sequentially with throttling"""
    results = {}
    for (chemo, pharma) in queries:
        results[(chemo, pharma)] = await run_async_query_test(chemo, pharma)
        await asyncio.sleep(1 / MAX_REQ_BY_SEC)


async def main_wait_for_creation(queries):
    """Create tasks sequentially with throttling"""
    coros = []
    for (chemo, pharma) in queries:
        coros.append(run_async_query_test(chemo, pharma))
        await asyncio.sleep(1 / MAX_REQ_BY_SEC)
    logger.info("main: %i jobs created", len(queries))

    for coro in asyncio.as_completed(coros):
        result = await coro
        logger.info(result)


async def main(queries):
    """Create tasks sequentially with throttling"""
    coros = []
    for i, (chemo, pharma) in enumerate(queries):
        delay = i * 1 / MAX_REQ_BY_SEC
        task = asyncio.create_task(run_async_query_test(chemo, pharma, delay=delay))
        coros.append(task)
    logger.info("main: %i jobs created", len(queries))

    res = await asyncio.gather(*coros)
    return {(chemo, pharma): (nb, duration) for (chemo, pharma, nb, duration) in res}


QUERIES = list(product(CHEMISTRY, ACTIVITIES))
NB_MAX_TEXT_QUERIES = 3

if __name__ == "__main__":
    main_start_time = datetime.datetime.now()
    logger.info("START")
    results = asyncio.run(main(sample(QUERIES, NB_MAX_TEXT_QUERIES)))
    pprint(results)
    total_time = datetime.datetime.now() - main_start_time
    logger.info("DONE in %fs", total_time.seconds + total_time.microseconds / 10 ** 6)

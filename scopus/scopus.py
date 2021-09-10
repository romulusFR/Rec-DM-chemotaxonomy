"""test de queries async à scopus via aoihttp"""

# %%

import logging
import asyncio
from itertools import product
import aiohttp
from random import randint

logging.basicConfig()
logger = logging.getLogger("scopus_api")
logger.setLevel(logging.DEBUG)

# https://www.twilio.com/blog/asynchronous-http-requests-in-python-with-aiohttp
# https://stackoverflow.com/questions/48682147/aiohttp-rate-limiting-parallel-requests
# https://docs.aiohttp.org/en/stable/client.html
# https://docs.aiohttp.org/en/stable/client_advanced.html


MAX_REQ_BY_SEC = 9
CHEMISTRY = ["alkaloid", "polyphenol", "coumarin"]
ACTIVITIES = ["antiinflammatory", "anticoagulant", "cancer"]


API_KEY = {"X-ELS-APIKey": "7047b3a8cf46d922d5d5ca71ff531b7d"}
search_and = lambda s1, s2: {"query": f"KEY({s1}) AND KEY({s2})", "count": 1}
x_rate_limit_headers = ["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"]


async def run_async_query_test(keyword1, keyword2):
    """launcher demo"""
    async with aiohttp.ClientSession() as session:
        url= "http://httpbin.org/anything"
        data = {"answer" : randint(1, 10000)}
        async with session.get(url, params=search_and(keyword1, keyword2), data=data) as resp:
            json = await resp.json()
            args = json["args"]["query"]
            results_nb = json["form"]["answer"]
            msg = f"(TEST) query={args}: {results_nb}"
            logger.info(msg)
            return int(results_nb)


async def run_async_query(keyword1, keyword2):
    """launcher"""
    async with aiohttp.ClientSession() as session:
        #scopus_url = "https://api.elsevier.com/content/search/scopus"
        scopus_url= "http://httpbin.org/anything"
        async with session.get(scopus_url, params=search_and(keyword1, keyword2), headers=API_KEY) as resp:
            logger.debug("X-RateLimit-Remaining=%s", resp.headers.get("X-RateLimit-Remaining", None))
            json = await resp.json()
            results_nb = json["search-results"]["opensearch:totalResults"]
            msg = f"{keyword1}.{keyword2}={results_nb}"
            logger.info(msg)
            return int(results_nb)


async def loop_wrap(queries):
    """Run **sequentially** to respect rate limits"""
    for query in queries:
        [chemo, pharma] = query
        # nb_results = await run_async_query(chemo, pharma)
        nb_results = await run_async_query_test(chemo, pharma)
        logger.info("Scopus has %i results for %s AND %s", nb_results, chemo, pharma)
        # yield (chemo, pharma, nb_results)
        await asyncio.sleep(1 / MAX_REQ_BY_SEC)


QUERIES = list(product(CHEMISTRY[:2], ACTIVITIES[:1]))


def main():

    loop = asyncio.get_event_loop()
    loop.run_until_complete(loop_wrap(QUERIES))
    # loop.run_forever()


if __name__ == "__main__":
    main()

"""test de queries async à scopus via aoihttp"""

# %%

import logging
import asyncio
from itertools import product
import aiohttp

logging.basicConfig()
logger = logging.getLogger("scopus_api")
logger.setLevel(logging.INFO)

# https://www.twilio.com/blog/asynchronous-http-requests-in-python-with-aiohttp
# https://stackoverflow.com/questions/48682147/aiohttp-rate-limiting-parallel-requests
# https://docs.aiohttp.org/en/stable/client.html
# https://docs.aiohttp.org/en/stable/client_advanced.html


MAX_REQ_BY_SEC = 9
CHEMISTRY = ["alkaloid", "polyphenol"]
ACTIVITIES = ["antiinflammatory", "anticoagulant", "cancer"]


API_KEY = {"X-ELS-APIKey": "7047b3a8cf46d922d5d5ca71ff531b7d"}
search_and = lambda s1, s2: {"query": f"KEY({s1}) AND KEY({s2})", "count": 1}
x_rate_limit_headers = ["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"]


async def run_async_query(keyword1, keyword2):
    """launcher"""
    async with aiohttp.ClientSession() as session:
        scopus_url = "https://api.elsevier.com/content/search/scopus"
        async with session.get(scopus_url, params=search_and(keyword1, keyword2), headers=API_KEY) as resp:
            for x_rate_limit in x_rate_limit_headers:
                logger.debug(resp.headers[x_rate_limit])

            json = await resp.json()

            results_nb = json["search-results"]["opensearch:totalResults"]
            msg = f"{keyword1}.{keyword2}={results_nb}"
            logger.info(msg)
            return msg


async def loop_wrap(queries):
    """Run parallel with rate limiter"""
    for query in queries:
        await run_async_query(*query)
        await asyncio.sleep(1 / MAX_REQ_BY_SEC)


QUERIES = list(product(CHEMISTRY, ACTIVITIES))

# %%
# asyncio.run(main())
if __name__ == "__main__":

    loop = asyncio.get_event_loop()
    loop.run_until_complete(loop_wrap(QUERIES))
    # loop.run_forever()

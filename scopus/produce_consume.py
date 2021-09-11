"""Producer/consumer for async tasks demo"""
# %%

import asyncio
import time
import math
import statistics
import logging
import pprint
from random import randint, lognormvariate

# from itertools import product

logging.basicConfig()
logger = logging.getLogger("asyncio_demo")
logger.setLevel(logging.INFO)

MU = 0.2
SIGMA = 1.0


async def produce(queue, queries):
    """Fill the queue with jobs"""
    for i, query in enumerate(queries):
        await queue.put(query)
        logger.debug("produce(): create job #%i q=%s.", i, query)


async def consume(queue, results, delay=1, name=None):
    """Take a job, do it, and wait at least 1sec in total, job duration included"""
    nb_jobs = 0
    while True:
        (kw1, kw2) = await queue.get()

        # models answer time from website
        # pause = randint(100, 500) / 1000
        duration = lognormvariate(MU, SIGMA)
        await asyncio.sleep(duration)
        res = randint(kw1, kw2)
        logger.info("consume(%s): got %s from job %s after %f", name, res, (kw1, kw2), duration)
        await asyncio.sleep(max(delay - duration, 0))
        nb_jobs += 1
        results.append((name, nb_jobs, duration))
        queue.task_done()


MAX_PER_SEC = 20
TOTAL_NB = 1000
QUERIES = [(i, i + 10) for i in range(TOTAL_NB)]


async def main(queries):
    """Launch all taks and wait for them until the job queue is empty"""
    jobs = asyncio.Queue()  # maxsize=MAX_PER_SEC
    results = []
    # on crée les deux tâches qui commencent à s'exécuter
    producer_task = asyncio.create_task(produce(jobs, queries), name="producer")
    consumer_tasks = [
        asyncio.create_task(consume(jobs, results, delay=1, name=i), name=f"consumer-{i}") for i in range(MAX_PER_SEC)
    ]

    logger.info("Tasks created")
    await asyncio.gather(producer_task)

    # on attend que tout soit traité, après que tout soit généré
    await jobs.join()
    logger.info("All jobs dones")
    for consumer in consumer_tasks:
        consumer.cancel()
    return results


# https://en.wikipedia.org/wiki/Log-normal_distribution
if __name__ == "__main__":
    now = time.perf_counter()
    logger.info("mean of lognormal(mu=%f, sigma=%f) %f", MU, SIGMA, math.exp(MU + (SIGMA ** 2) / 2))
    times = asyncio.run(main(QUERIES))
    logger.info("elapsed %f", time.perf_counter() - now)
    mean = statistics.fmean(d for (j, n, d) in times)

    logger.info("mean duration %f", mean)
    logger.info("\n %s", pprint.pformat(times))

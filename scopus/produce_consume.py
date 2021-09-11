# %%

import asyncio
import time
import logging
from random import randint
from itertools import product

logging.basicConfig()
logger = logging.getLogger("asyncio_demo")
logger.setLevel(logging.DEBUG)

TOTAL_NB = 20
queries = [(i, i + 10) for i in range(TOTAL_NB)]


async def produce(queue, delay=0):
    for i, query in enumerate(queries):
        await queue.put(query)
        logger.debug("produce(): create job #%i q=%s.", i, query)
        await asyncio.sleep(delay)


async def consume(queue, delay=0, name=None):
    while True:
        (kw1, kw2) = await queue.get()

        pause = randint(100, 500) / 1000
        await asyncio.sleep(pause)
        res = randint(kw1, kw2)
        logger.info("consume(%s): got %s from job %s after %f", name, res, (kw1, kw2), pause)
        await asyncio.sleep(delay - pause)
        queue.task_done()


MAX_PER_SEC = 10


async def main():
    jobs = asyncio.Queue() # maxsize=MAX_PER_SEC

    # on crée les deux tâches qui commencent à s'exécuter
    producer_task = asyncio.create_task(produce(jobs, delay=0), name="producer")
    consumer_tasks = [asyncio.create_task(consume(jobs, delay=1, name=i), name=f"consumer-{i}") for i in range(MAX_PER_SEC)]

    logger.info("Tasks created")
    await asyncio.gather(producer_task)

    # on attend que tout soit traité, après que tout soit généré
    await jobs.join()
    logger.info("All jobs dones")
    for consumer in consumer_tasks:
        consumer.cancel()


if __name__ == "__main__":
    now = time.perf_counter()
    asyncio.run(main())
    logger.info("elapsed %f", time.perf_counter() - now)

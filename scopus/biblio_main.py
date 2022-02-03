"""CLI for the bibliographical extractor"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from pprint import pformat

import biblio_extractor as bex

logging.basicConfig()
logger = logging.getLogger("CHEMOTAXO")

# Output
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_parser() -> argparse.ArgumentParser:
    """argparse configuration"""
    arg_parser = argparse.ArgumentParser(description="Scopus downloader")
    arg_parser.add_argument(
        "filename",
        help="file to read chemo activities from",
    )
    arg_parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="verbosity level, default is WARNING (30). Use -v once for INFO (20) and twice -vv for DEBUG (10).",
    )
    arg_parser.add_argument(
        "--search",
        "-sm",
        action="store",
        default=bex.DEFAULT_SEARCH_MODE,
        help=f"search mode: 'offline', 'fake', 'httpbin' or 'scopus' (default '{bex.DEFAULT_SEARCH_MODE}')",
    )
    arg_parser.add_argument(
        "--parallel",
        "-p",
        type=int,
        action="store",
        default=bex.DEFAULT_PARALLEL_WORKERS,
        help=f"number of parallel consumers/workers (default {bex.DEFAULT_PARALLEL_WORKERS})",
    )
    arg_parser.add_argument(
        "--delay",
        "-d",
        type=float,
        action="store",
        default=bex.DEFAULT_WORKER_DELAY,
        help=f"minimum delay between two consecutive queries from a worker (default {bex.DEFAULT_WORKER_DELAY})",
    )
    arg_parser.add_argument(
        "--samples",
        "-s",
        type=int,
        action="store",
        default=bex.DEFAULT_SAMPLES,
        help="maximum number of queries (random samples) (default all queries)",
    )
    arg_parser.add_argument(
        "--write",
        "-w",
        action="store_true",
        default=False,
        help="writes results to csv file (default False)",
    )
    arg_parser.add_argument(
        "--margins",
        "-m",
        action="store_true",
        default=False,
        help="also queries and returns marginal sums",
    )
    return arg_parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # https://docs.python.org/3/library/logging.html#levels
    if args.verbose >= 2:
        LEVEL = logging.DEBUG
    elif args.verbose == 1:
        LEVEL = logging.INFO
    else:
        LEVEL = logging.WARNING
    logger.setLevel(LEVEL)

    print(f"Scopus downloader started (debug={logger.getEffectiveLevel()})")
    logger.debug(pformat(vars(args)))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("output dir is '%s'", OUTPUT_DIR.absolute())
    logger.info("Scopus API key %s", bex.API_KEY)

    filename = Path(args.filename)
    dataset = bex.load_data(filename)
    all_compounds = list(dataset.index.get_level_values(1))
    all_activities = list(dataset.columns.get_level_values(1))
    print(f"Loaded {len(all_compounds)} compounds and {len(all_activities)} activities")
    logger.info("all compounds %s", all_compounds)
    logger.info("all activities %s", all_activities)

    # dataset = load_chemo_activities(args.filename)

    if args.search not in bex.SEARCH_MODES:
        raise ValueError(f"Unknown search mode {args.search}")

    if args.search == "offline":
        nb_papers = 243964 // (len(all_compounds) * len(all_activities))
        results = bex.gen_db(list(dataset.index), list(dataset.columns), nb_papers , 383330 / 243964)

    else:
        nb_queries = (
            args.samples
            if args.samples is not None
            else 4 * len(all_compounds) * len(all_activities)
            + (2 * len(all_compounds) + 2 * len(all_activities) + 1) * args.margins
        )
        print(
            f"Launching {nb_queries} queries using {args.search} with {args.parallel} parallel workers (w/ min delay {args.delay})"
        )

        results = bex.launcher(
            dataset,
            task_factory=bex.SEARCH_MODES[args.search],
            with_margin=args.margins,
            parallel_workers=args.parallel,
            worker_delay=args.delay,
            samples=args.samples,
        )

    print(results)

    if args.write:
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_filename = OUTPUT_DIR / f"{filename.stem}_{now}.csv"
        results.to_csv(output_filename)
        logger.info("results written to %s", output_filename)

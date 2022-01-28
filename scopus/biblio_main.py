import argparse
import logging
from pprint import pformat

import biblio_extractor as bl_ex

logging.basicConfig()
logger = logging.getLogger("CHEMOTAXO")

# DEFAULT VALUES
DEFAULT_WORKERS = 8
DEFAULT_DELAY_PER_WORKER = 1.0
DEFAULT_SAMPLES = None
DEFAULT_MODE = "fake"


def get_parser():
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
    if args.verbose >= 2:
        LEVEL = logging.DEBUG
    elif args.verbose == 1:
        LEVEL = logging.INFO
    else:
        LEVEL = logging.WARNING
    logger.setLevel(LEVEL)

    print(f"Scopus downloader started (debug={logger.getEffectiveLevel()})")
    logger.debug(pformat(vars(args)))

    dataset = bl_ex.load_data(args.filename)
    print(dataset.to_string())
    # dataset = load_chemo_activities(args.filename)
    # download_all(
    #     dataset,
    #     mode=args.mode,
    #     parallel=args.parallel,
    #     samples=args.samples,
    #     no_write=args.no_write,
    #     delay=args.delay,
    # )
    # pass
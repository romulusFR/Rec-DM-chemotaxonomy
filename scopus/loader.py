# pylint: disable=unused-import
# %%
"""I/O for chemo activities of natural compounds"""

import csv
import logging
from pathlib import Path
from pprint import pprint, pformat
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


logging.basicConfig()
logger = logging.getLogger("scopus_api")
logger.setLevel(logging.INFO)

CSV_PARAMS = {"delimiter": ";", "quotechar": '"'}
BASE_CLASS = "*"
NO_DATA = None

# TODO : see if pandas with both hierarchical indexes and attributes is ok
@dataclass
class Dataset:
    """Data structure to store cehmo activities

    Stores :
        - chemical compounds with their classes
        - pharmacological/biological activities with their classes
        - the number of publications about both compound and activity in a dict
    """

    compounds: dict[str, str]
    activities: dict[str, str]
    data: dict[str, dict[str, Optional[int]]]

    def to_numpy(self) -> np.ndarray:
        """Converts the data part to a numpy array"""
        np_data: np.ndarray = np.ndarray(shape=(len(self.compounds), len(self.activities)))
        for compound_idx, compound in enumerate(self.compounds):
            for activity_idx, activity in enumerate(self.activities):
                np_data[compound_idx][activity_idx] = self.data[compound][activity]
        return np_data


def standardize(string: str) -> str:
    """Standardize strings"""
    if not string:
        return BASE_CLASS
    return string.strip().lower()


def load_chemo_activities(filename: Path):
    """Loads chemical compounds / pharmacological activities matrice"""
    # number of columns (containing compounds) to skip
    skip_cols = 2
    logger.debug("load_chemo_activities('%s')", filename)
    with open(filename, "r", newline="", encoding="utf-8") as csvfile:
        csvreader = csv.reader(csvfile, **CSV_PARAMS)
        # 1st line contains pharmacological/biological classes
        pharma_classes = list(map(standardize, next(csvreader)[skip_cols:]))
        # 2nd line contains pharmacological/biological effects
        pharma_effects = list(map(standardize, next(csvreader)[skip_cols:]))
        # put {effect : classes} into a dict
        pharma_dict: dict[str, str] = dict(zip(pharma_effects, pharma_classes))

        logger.debug(pharma_dict)

        chemo_dict: dict[str, str] = {}
        data_dict: dict[str, dict[str, Optional[int]]] = {}
        # now, read all subsequent lines
        for row in csvreader:
            # 1st column is chemical class, 2nd is compound
            chemo_class = standardize(row[0])
            chemo_compound = standardize(row[1])
            chemo_dict[chemo_compound] = chemo_class
            data_dict[chemo_compound] = {}
            for idx, cell in enumerate(row[skip_cols:]):
                val: Optional[int] = None
                try:
                    val = int(cell, base=10)
                except ValueError:
                    val = NO_DATA
                data_dict[chemo_compound][pharma_effects[idx]] = val

        logger.debug(chemo_dict)
        logger.debug(pformat(data_dict))
        logger.info("load_chemo_activities('%s') loaded", filename)
        return Dataset(chemo_dict, pharma_dict, data_dict)


def sort_by_class(classes):
    """Ensure that parents classes are first in order, and that children are classes-wise ordered alphabeticaly"""
    return sorted((com for com, cls in classes.items()), key=lambda x: (classes[x], x))


def write_chemo_activities(filename: Path, dataset: Dataset):
    """Store result dict"""
    logger.debug("write_chemo_activities(%s)", filename)
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, **CSV_PARAMS)
        # pharmacological activity classes
        activities = sort_by_class(dataset.activities)
        writer.writerow(("", "", *(dataset.activities[col] for col in activities)))
        # pharmacological activities
        writer.writerow(("", "", *activities))
        # write compounds, one by line
        compounds = sort_by_class(dataset.compounds)
        for compound in compounds:
            values = [dataset.data[compound].get(col, NO_DATA) for col in activities]
            writer.writerow((dataset.compounds[compound], compound, *values))
    logger.info("write_chemo_activities(%s) written", filename)

# test : loads and then rewrites

if __name__ == "__main__":
    INPUT_DIR = Path("data")
    OUTPUT_DIR = Path("results")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHEMO_ACTIVITIES_FILENAME = INPUT_DIR / "activities.csv"

    logger.info("Start at %s", datetime.now())
    chemo_activities = load_chemo_activities(CHEMO_ACTIVITIES_FILENAME)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = OUTPUT_DIR / f"activity_{now}.csv"
    write_chemo_activities(output_filename, chemo_activities)
    logger.info("Done at %s", datetime.now())

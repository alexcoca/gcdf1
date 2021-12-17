"""A module containing iterators for the SGD corpus and various key functions that can
be used for sorting.
"""
from __future__ import annotations

import itertools
import json
import logging
import pathlib
import re
from collections import defaultdict
from typing import List, Literal, Optional, Union

try:
    import importlib_resources
except ImportError:
    pass
import numpy as np

_DATA_PACKAGE = "data.raw"


class PathMapping:

    split_names = ["train", "dev", "test"]

    def __init__(self, data_pckg_or_path: Union[str, pathlib.Path] = _DATA_PACKAGE):
        self.pckg = data_pckg_or_path
        try:
            self.data_root = importlib_resources.files(data_pckg_or_path)
        except (ModuleNotFoundError, NameError, TypeError):
            if isinstance(data_pckg_or_path, str):
                self.data_root = pathlib.Path(data_pckg_or_path)
        self._all_files = [r for r in self.data_root.iterdir()]
        self.split_paths = self._split_paths()
        self.schema_paths = self._schema_paths()

    def _split_paths(self):
        paths = {}
        for split in PathMapping.split_names:
            r = [f for f in self._all_files if f.name == split]
            if not r:
                continue
            [paths[split]] = r
        return paths

    def _schema_paths(self):
        return {
            split: self.split_paths[split].joinpath("schema.json")
            for split in PathMapping.split_names
            if split in self.split_paths
        }

    def _get_split_path(self, split: Literal["train", "test", "dev"]) -> pathlib.Path:
        return self.split_paths[split]

    def _get_schema_path(self, split: Literal["train", "test", "dev"]) -> pathlib.Path:
        return self.schema_paths[split]

    def __getitem__(self, item):
        if item in PathMapping.split_names:
            return self.split_paths[item]
        else:
            if item != "schema":
                raise ValueError(
                    f"Keys available are schema and {*PathMapping.split_names,}"
                )
            return self.schema_paths


def reconstruct_filename(dial_id: str) -> str:
    """Reconstruct filename from dialogue ID."""

    file_prefix = int(dial_id.split("_")[0])

    if file_prefix in range(10):
        str_file_prefix = f"00{file_prefix}"
    elif file_prefix in range(10, 100):
        str_file_prefix = f"0{file_prefix}"
    else:
        str_file_prefix = f"{file_prefix}"

    return f"dialogues_{str_file_prefix}.json"


def get_file_map(
    dialogue_ids: list[str],
    split: Literal["train", "test", "dev"],
    data_pckg_or_path: str = "data.raw",
) -> dict[pathlib.Path, list[str]]:
    """Returns a map where the keys are file paths and values are lists
    comprising dialogues from `dialogue_ids` that are in the same file.

    dialogue_ids:
        IDs of the dialogues whose paths are to be returned, formated as the schema 'dialogue_id' field.
    split:
        The name of the split whose paths are to be returned.
    data_pckg_or_path:
        The location of the python package where the data is located
    """

    file_map = defaultdict(list)
    path_map = PathMapping(data_pckg_or_path=data_pckg_or_path)
    for id in dialogue_ids:
        # ValueError occurs if dialogue IDs do not match SGD convention
        try:
            fpath = path_map[split].joinpath(reconstruct_filename(id))
        except ValueError:
            found_dialogue = False
        else:
            # for the original SGD data, one can reconstruct the filename
            # from dial ID to load the dialogue
            file_map[fpath].append(id)
            continue
        # in general, just iterate through the file to find a given
        # dialogue
        if not found_dialogue:
            for fpath in path_map[split].iterdir():
                if not fpath.name.startswith("dialogues"):
                    continue
                with open(fpath, "r") as f:
                    dial_bunch = json.load(f)
                for dial in dial_bunch:

                    if dial["dialogue_id"] == id:
                        found_dialogue = True
                        break

                if found_dialogue:
                    break

            if found_dialogue:
                file_map[fpath].append(id)
            else:
                logging.warning(f"Could not find dialogue {id}...")

    return file_map


def get_filepaths(
    split: Literal["train", "test", "dev"], data_pckg_or_path: str = "data.raw"
) -> list[pathlib.Path]:
    """Returns a list of file paths for all dialogue batches in a given split.

    Parameters
    ----------
    split
        The split whose filepaths should be returned
    data_pckg_or_path
        The package where the data is located.
    """
    path_map = PathMapping(data_pckg_or_path=data_pckg_or_path)
    fpaths = list(path_map[split].glob("dialogues_*.json"))
    if "dialogues_and_metrics.json" in fpaths:
        fpaths.remove("dialogues_and_metrics.json")
    return fpaths


def file_iterator(
    fpath: pathlib.Path, return_only: Optional[set[str]] = None
) -> tuple[str, dict]:
    """
    Iterator through an SGD .json file.

    Parameters
    ----------
    fpath:
        Absolute path to the file.
    return_only
        A set of dialogues to be returned. Specified by dialogue IDs as
        found in the `dialogue_id` file of the schema.
    """

    with open(fpath, "r") as f:
        dial_bunch = json.load(f)

    n_dialogues = len(dial_bunch)
    try:
        max_index = int(dial_bunch[-1]["dialogue_id"].split("_")[1]) + 1
    except IndexError:
        max_index = -100
    missing_dialogues = not (max_index == n_dialogues)

    if return_only:
        if not missing_dialogues:
            for dial_idx in (int(dial_id.split("_")[1]) for dial_id in return_only):
                yield fpath, dial_bunch[dial_idx]
        else:
            returned = set()
            for dial in dial_bunch:
                found_id = dial["dialogue_id"]
                if found_id in return_only:
                    returned.add(found_id)
                    yield fpath, dial
                    if returned == return_only:
                        break
            if returned != return_only:
                logging.warning(f"Could not find dialogues: {return_only - returned}")
    else:
        for dial in dial_bunch:
            yield fpath, dial


def split_iterator(
    split: Literal["train", "dev", "test"],
    return_only: Optional[set[str]] = None,
    data_pckg_or_path: str = "data.raw",
) -> tuple[pathlib.Path, dict]:
    """

    Parameters
    ----------
    split
        Split through which to iterate.
    return_only
        Return only certain dialogues, specified by their schema ``dialogue_id`` field.
    data_pckg_or_path
        Package where the data is located.
    """
    # return specified dialogues only
    if return_only:
        fpath_map = get_file_map(
            list(return_only), split, data_pckg_or_path=data_pckg_or_path
        )
        for fpth, dial_ids in fpath_map.items():
            yield from file_iterator(fpth, return_only=set(dial_ids))
    # iterate through all dialogues
    else:
        for fp in get_filepaths(split, data_pckg_or_path=data_pckg_or_path):
            with open(fp, "r") as f:
                dial_bunch = json.load(f)
            for dial in dial_bunch:
                yield fp, dial


def corpus_iterator(data_pckg_or_path: str = "data.raw"):

    path_map = PathMapping(data_pckg_or_path=data_pckg_or_path)
    for split in path_map.split_names:
        yield from split_iterator(split, data_pckg_or_path=data_pckg_or_path)


def dialogue_iterator(dialogue: dict, user: bool = True, system: bool = True) -> dict:

    if (not user) and (not system):
        raise ValueError("At least a speaker needs to be specified!")

    filter = "USER" if not user else "SYSTEM" if not system else ""

    for turn in dialogue["turns"]:
        if filter and turn.get("speaker", "") == filter:
            continue
        else:
            yield turn


def actions_iterator(frame: dict, patterns: Optional[List[str]] = None) -> dict:
    """
    Iterate through actions in a frame.

    Parameters
    ----------
    patterns
        If supplied, only actions whose ``act`` field is matched by at least one pattern are returned.
    """

    for act_dict in frame.get("actions", []):
        if patterns:
            for pattern in patterns:
                if re.search(pattern, act_dict["act"]):
                    yield act_dict
        else:
            yield act_dict


def schema_iterator(
    split: Literal["train", "test", "dev"], package: str = "data.raw"
) -> dict:

    path_mapping = PathMapping(data_pckg_or_path=package)
    with open(path_mapping["schema"][split], "r") as f:
        schema = json.load(f)
    for service in schema:
        yield service


def random_sampler(
    split: Literal["train", "test", "dev"], n_samples: int, data_pckg: str = "data.raw"
) -> list[tuple[pathlib.Path, dict]]:

    iterator = split_iterator(split, data_pckg_or_path=data_pckg)
    reservoir = list(itertools.islice(iterator, n_samples))

    for idx, elem in enumerate(iterator, n_samples + 1):
        i = np.random.randint(1, idx + 1)

        if i <= n_samples:
            reservoir[i - 1] = elem

    return reservoir


def get_turn_by_idx(dialogue: dict, idx: int) -> dict:
    return dialogue["turns"][idx]


def dial_sort_key(dialogue_id: str) -> tuple[int, int]:
    s1, s2 = dialogue_id.split("_")
    return int(s1), int(s2)


def alphabetical_sort_key(name: str, n_chars: int = 10) -> str:
    return name[:n_chars]


def dial_files_sort_key(name: str) -> int:
    return int(name.split("_")[1].split(".")[0])

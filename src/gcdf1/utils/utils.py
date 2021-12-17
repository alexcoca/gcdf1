from __future__ import annotations

import subprocess
from collections import Callable, defaultdict
from copy import deepcopy
from datetime import datetime
from functools import partial
from itertools import repeat
from typing import Union


def cast_vals_to_set(d: dict):
    """Maps the values of a nested dictionary to a set of strings."""

    for key, value in d.items():
        if isinstance(value, dict):
            cast_vals_to_set(value)
        else:
            d[key] = set(value)
    return d


def get_commit_hash():
    """Returns the commit hash for the current HEAD."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()


def get_datetime() -> str:
    """Returns the current date and time."""
    now = datetime.now()
    return now.strftime("%d/%m/%Y %H:%M:%S")


def append_to_values(result: dict, new_data: dict):
    """Recursively appends to the values of `result` the values in
    `new_data` that have the same keys. If the keys in `new_data`
    do not exist in `result`, they are recursively added. The keys of
    `new_data` can be either lists or single float objects that
    are to be appended to existing `result` keys. List concatenation is
    performed in former case.

    Parameters
    ----------
    result
        Mapping whose values are to be extended with corresponding values from
        `new_data_map`
    new_data
        Data with which the values of `result_map` are extended.
    """
    for key in new_data:
        # recursively add any new keys to the result mapping
        if key not in result:
            if isinstance(new_data[key], dict):
                result[key] = dict_factory()
                append_to_values(result[key], new_data[key])
            else:
                if isinstance(new_data[key], float):
                    result[key] = [new_data[key]]
                elif isinstance(new_data[key], list):
                    result[key] = [*new_data[key]]
                else:
                    raise ValueError("Unexpected key type.")
        # updated existing values with the value present in `new_data_map`
        else:
            if isinstance(result[key], dict):
                append_to_values(result[key], new_data[key])
            else:
                if isinstance(new_data[key], list):
                    result[key] += new_data[key]
                elif isinstance(new_data[key], float):
                    result[key].append(new_data[key])
                else:
                    raise ValueError("Unexpected key type")


def dict_factory():
    return defaultdict(list)


def get_nested_values(mapping: dict, nested_keys: tuple[Union[list, tuple], ...]):
    """Returns a list of values for nested keys. A nested key is specified as a list."""
    vals = []
    # key is a nested key in a dict represented as a list
    for key in nested_keys:
        new = {}
        i = 0
        while i < len(key):
            this_level_key = key[i]
            if not new:
                if this_level_key in mapping:
                    new = mapping[this_level_key]
                else:
                    break
            else:
                new = new[this_level_key]
            i += 1
        vals.append(new)
    return vals


def nested_defaultdict(default_factory: Callable, depth: int = 1) -> defaultdict:
    """Creates a nested default dictionary of arbitrary depth with a specified callable as leaf."""
    if not depth:
        return default_factory()
    result = partial(defaultdict, default_factory)
    for _ in repeat(None, depth - 1):
        result = partial(defaultdict, result)
    return result()


def safeget(dct: dict, *keys: Union[tuple[str], list[str]]):
    """Retrieves the value of one nested key represented in `keys`"""
    for key in keys:
        try:
            dct = dct[key]
        except KeyError:
            return None
    return dct


def dispatch_on_value(func: Callable) -> Callable:
    """
    Value-dispatch function decorator.

    Transforms a function into a value-dispatch function,
    which can have different behaviors based on the value of the first argument.
    """

    registry = {}

    def dispatch(value):

        try:
            return registry[value]
        except KeyError:
            return func

    def register(value, func=None):
        def add_to_register(func):
            register(value, func)

        if func is None:
            return add_to_register

        registry[value] = func

        return func

    def wrapper(*args, **kw):
        return dispatch(args[0])(*args, **kw)

    wrapper.register = register
    wrapper.dispatch = dispatch
    wrapper.registry = registry

    return wrapper


def count_odd(left: int, right: int) -> int:
    """Counts the number of odd numbers.

    left,right
        Interval boundaries.
    """
    n_odd = (right - left) // 2

    if right % 2 != 0 or left % 2 != 0:
        n_odd += 1

    return n_odd


def count_nested_dict_values(d: dict) -> dict:
    """Returns a mapping with the same structure as the input but with the values replaced by the counts of the
    value fields."""

    count_d = deepcopy(d)

    def _helper(d: dict, count_d: dict):

        for key, value in d.items():
            if isinstance(value, dict):
                _helper(value, count_d[key])
            else:
                count_d[key] = len(value)

    _helper(d, count_d)

    return count_d


def count_floats(param: dict) -> int:
    """Count how may floats there are in a nested dictionary."""
    n_floats = 0
    for key, value in param.items():
        if isinstance(value, dict):
            n_floats += count_floats(value)
        else:
            if not hasattr(value, "__iter__"):
                raise ValueError(f"Dictionary had non-iterable value: {value}")
            assert all(isinstance(el, float) for el in value)
            n_floats += len(value)

    return n_floats


def sum_nested_dict_values(d: dict):
    """Returns the sum of the values in a nested dictionary."""

    # TODO: TEST THIS FUNCTION

    current_sum = 0
    for key, value in d.items():
        if isinstance(value, dict):
            current_sum += sum_nested_dict_values(value)
        else:
            current_sum += value

    return current_sum


def default_to_regular(d: defaultdict) -> dict:
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d


def store_data(data, store: defaultdict, store_fields: list[str]):
    """Stores data in a nested default dictionary at a specified location.

    Parameters
    ----------
    data
        Data to be stored.
    store
        Nested default dictionary where data is to be stored.
    store_fields
        Key where the data is to be stored

    Notes
    -----
    Supported types for `store` leaves are ``int`` and ``list``.
    """

    store_location = safeget(store, *store_fields)
    if isinstance(store_location, list):
        store_location.append(data)
    elif isinstance(store_location, int):
        store_location = safeget(store, *store_fields[:-1])
        if isinstance(data, int):
            if isinstance(store_location, defaultdict):
                if isinstance(store_location[store_fields[-1]], int):
                    store_location[store_fields[-1]] += data
                else:
                    raise NotImplementedError
    else:
        raise NotImplementedError

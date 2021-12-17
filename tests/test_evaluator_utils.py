# format input as pairs of mapping and all nested keys within
from collections import OrderedDict
from copy import deepcopy
from itertools import chain
from math import prod

import pytest

from gcdf1.utils.evaluator import (
    aggregate_values,
    get_metrics_tracker,
    metrics_dict_factory,
)
from gcdf1.utils.utils import append_to_values, get_nested_values

test_data = [
    (
        OrderedDict({"a": [1.0, 2.0, 3.0], "b": {"c": {"d": [1.0, 2.2]}}}),
        (["a"], ["b", "c", "d"]),
    ),
    (OrderedDict({"a": [2.0, 3.0]}), (["a"])),
    (OrderedDict({"a": [2.0], "b": [3.0], "c": [4.0]}), ["a", "b", "c"]),
    (
        OrderedDict({"a": [2.0], "b": [3.0], "c": {"d": [4.0, 6.0]}}),
        (["a"], ["b"], ["c", "d"]),
    ),
]
agg_fcn = ["mean", "prod"]


@pytest.mark.parametrize("test_data", test_data, ids="test_data={}".format)
@pytest.mark.parametrize("agg_fcn", agg_fcn, ids="agg_fcn={}".format)
def test_aggregate_values(test_data, agg_fcn):

    mapping, nested_keys = test_data
    mapping = deepcopy(mapping)
    if agg_fcn == "mean":
        expected_vals = [
            sum(res) / len(res) for res in get_nested_values(mapping, nested_keys)
        ]
    elif agg_fcn == "prod":
        expected_vals = [prod(res) for res in get_nested_values(mapping, nested_keys)]

    aggregate_values(mapping, agg_fcn)  # noqa
    actual_values = get_nested_values(mapping, nested_keys)
    assert actual_values == expected_vals


base_dict = OrderedDict(
    {
        "a": [1.0],
        "b": {"c": [1.0]},
    }
)
update_dict = OrderedDict(
    {
        "a": 2.0,
        "b": {"c": 3.0},
        "d": 4.0,
        "e": {"f": 5.0},
    }
)
new_keys_vals = ((["d"], 4.0), (["e", "f"], 5.0))
new_vals = ((["a"], 2.0), (["b", "c"], 3.0))


@pytest.mark.parametrize(
    "base",
    [
        base_dict,
    ],
    ids="base={}".format,
)
@pytest.mark.parametrize(
    "update",
    [
        (update_dict, new_keys_vals, new_vals),
    ],
    ids="update={}".format,
)
def test_append_to_values(base, update):

    update_dict, new_keys_vals, new_vals = update
    append_to_values(base, update_dict)
    for nested_key_val in chain(new_keys_vals, new_vals):
        nested_key, val = nested_key_val
        ret_val = get_nested_values(base, (nested_key,))[0]
        assert ret_val[-1] == val


dummy_metrics = ["a", "b", "c"]


@pytest.mark.parametrize(
    "dummy_metrics",
    [
        dummy_metrics,
    ],
    ids="dummy_metrics={}".format,
)
def test_dialogue_metrics_tracker(dummy_metrics):

    tracker = get_metrics_tracker(dummy_metrics)()
    assert tracker.metrics() == dummy_metrics
    expected_type = type(metrics_dict_factory())
    for metric in tracker.metrics():
        assert type(getattr(tracker, metric)) is expected_type

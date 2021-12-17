from __future__ import annotations

import glob
import json
from collections import Counter, defaultdict, namedtuple
from copy import deepcopy
from dataclasses import dataclass, field, fields, make_dataclass
from operator import methodcaller
from typing import Callable, Literal, Optional, Union

import numpy as np
import prettyprinter
from absl import logging
from fuzzywuzzy import fuzz
from nltk.util import ngrams

from gcdf1.utils.utils import (
    append_to_values,
    count_nested_dict_values,
    get_nested_values,
)

_METRIC_NAMES = ["f1", "precision", "recall"]
F1Scores = namedtuple("F1Scores", _METRIC_NAMES)


def aggregate_values(mapping: dict, agg_fcn: Literal["mean", "prod"]):
    """Aggregates the values of the input (nested) mapping according to the
    specified aggregation method. This function modifies the input in place.

    Parameters
    ---------
    mapping
        The mapping to be aggregated.
    agg_fcn
        Aggregation function. Only  `mean` or `prod` aggregation supported.
    """

    for key, value in mapping.items():
        if isinstance(value, dict):
            aggregate_values(mapping[key], agg_fcn)
        else:
            aggregator = methodcaller(agg_fcn, value)
            mapping[key] = aggregator(np)


def metrics_dict_factory():
    """Factory function that returns a two level nested dictionary for storing metric values."""
    return defaultdict(lambda: defaultdict(list))


def asdict(tracker):
    """Function to convert metrics tracker to dict. dataclasses.asdict() fails
    because the metrics tracker has collections.defaultdict objects as fields."""

    converted = {}
    for metric in tracker.metrics():
        converted[metric] = getattr(tracker, metric)

    return converted


@dataclass
class MetricsTracker:

    metadata_to_agg = ()

    def __iadd__(self, other):
        """In place addition operator for the tracker."""
        other_metrics = other.metrics()
        self_metrics = self.metrics()
        if other_metrics != self_metrics:
            if set(other_metrics).symmetric_difference(self_metrics) != {"metadata"}:
                raise ValueError("Metrics must match for accumulation!")
        for metric in self_metrics:
            if metric == "metadata":
                for key in self.metadata_to_agg:
                    current_info = get_nested_values(getattr(self, metric), (key,))[0]
                    new_info = get_nested_values(getattr(other, metric), (key,))[0]
                    assert isinstance(new_info, dict)
                    if new_info:
                        append_to_values(current_info, new_info)
                continue
            # current dialogue might now have new values for all metrics
            # (e.g., entity score not measured for transaction only dialogues)
            new_values = getattr(other, metric)
            if new_values:
                if isinstance(new_values, dict):
                    append_to_values(getattr(self, metric), new_values)
                elif isinstance(new_values, list):
                    old_values = getattr(self, metric)
                    old_values += new_values
                else:
                    raise TypeError(
                        f"In place addition not defined for type {type(new_values)}"
                    )
        return self

    def __str__(self):
        pretty_output = []
        for metric in self.metrics():
            metric_vals = getattr(self, metric, {})
            pretty_format = prettyprinter.pformat(dict(metric_vals))
            pretty_output.append(f"metric: {metric}\n{pretty_format}")
        return "\n".join(pretty_output)

    def metrics(self):
        """Returns a list of all metrics the tracker can track."""
        return [field.name for field in fields(self)]

    def tracked_metrics(self):
        """Returns a list of metrics that have been computed for the current dialogue."""
        all_metrics = self.metrics()
        return [metric for metric in all_metrics if getattr(self, metric)]

    def aggregate_metrics(self, agg_fcn: Literal["mean", "prod"]):
        # metric might not be recorded (e.g., entity score is
        # not computed in transactional only dialogues)
        if hasattr(self, "metadata"):
            self._add_score_distributions()

        for metric in self.tracked_metrics():
            if metric == "metadata":
                continue
            values = getattr(self, metric)
            if isinstance(values, dict):
                aggregate_values(getattr(self, metric), agg_fcn=agg_fcn)
            else:
                setattr(
                    self,
                    metric,
                    np.mean(values) if agg_fcn == "mean" else np.prod(values),
                )

    def _add_score_distributions(self):
        # TODO: REFACTOR METADATA STORE SO THAT THIS BECOMES INDEPENDENT OF METRICS.
        """Stores the scores distributions and the counts of dialogues per domain."""
        metadata_dict = getattr(self, "metadata")
        metadata_dict["score_distributions"] = {}
        metadata_dict["counts"] = {}
        for metric in self.tracked_metrics():
            if metric not in metadata_dict:
                continue
            dial_id_store = metadata_dict[metric]["dial_id"]
            if metric == "metadata":
                continue
            metadata_dict["counts"][metric] = deepcopy(
                count_nested_dict_values(getattr(self, metric))
            )
            metadata_dict["score_distributions"][metric] = deepcopy(
                getattr(self, metric)
            )
            combined_distribution = []
            for domain_key in metadata_dict["score_distributions"][metric]:
                domain_distribution = metadata_dict["score_distributions"][metric][
                    domain_key
                ]
                domain_count = metadata_dict["counts"][metric][domain_key]
                assert len(domain_distribution) == domain_count
                domain_dial_id = dial_id_store[domain_key]
                assert len(domain_distribution) == len(domain_dial_id)
                new_distribution = list(zip(domain_distribution, domain_dial_id))
                metadata_dict["score_distributions"][metric][
                    domain_key
                ] = new_distribution
                combined_distribution += new_distribution
            metadata_dict["score_distributions"][metric][
                "combined"
            ] = combined_distribution

        del metadata_dict["igcd_f1"]["dial_id"]
        del metadata_dict["rgcd_f1"]["dial_id"]


def get_metrics_tracker(
    metrics: list[str],
    metrics_store_factory: Optional[Union[dict[str, Callable], Callable]] = None,
    metadata: bool = False,
    metadata_store_factory: Optional[Callable] = None,
    metadata_to_agg: tuple[list[str], ...] = (),
):
    """Initialises a metrics tracker across dialogues.

    Parameters
    ----------
    metrics
        The names of the metrics to be tracked.
    metrics_store_factory
        The container where the metrics are stored. The following behaviour is implemented given type:

            -``None``: the container for each is by default::

                {
                "key_1": "key_1_1": [],
                ...
                }

            - ``Callable``: each metric will have the container returned by calling the callable with no args. The
                callable should return a `dict` instance.

            - ``dict``: the keys must be the same as the metrics. Values must be either ``Callable`` or `None` \
            and each metric is assigned a container based on the type of the value
    metadata
        If True, a metadata field is added to the tracker. This is ignored during aggregation.
    metadata_store_factory
        If not specified, the metadata field is a dictionary. Otherwise, the metadata will be stored in the object
        returned by calling  this function with no arguments. The factory should return an instance of dict.
    metadata_to_agg:
        This tuple should contain lists of coma separated strings, indicating which fields in the underlying data store
        are to be aggregated. See gcdf1.utils.multiwoz_output for an example. This argument is necessary to ensure the
        hierarchical output is extended as each dialogue is evaluated via the MetricTracker's __iadd__ method.
    """

    if not isinstance(metrics_store_factory, dict):
        metrics_container = (
            metrics_store_factory if metrics_store_factory else metrics_dict_factory
        )
        fields = [
            (
                metric,
                type(metrics_container()),
                field(default_factory=metrics_container),
            )
            for metric in metrics
        ]

    else:
        if set(metrics) - metrics_store_factory.keys():
            raise ValueError(
                f"A default factory must be specified per metric if metrics_store_factory is dict. Got {set(metrics)}"
                f" but metrics_store_factory dict had keys {set(metrics_store_factory.keys())}"
            )
        fields = []
        for metric in metrics:
            fields.append(
                (
                    metric,
                    type(metrics_store_factory[metric]()),
                    field(default_factory=metrics_store_factory[metric]),
                )
            )
    if metadata:
        meta_default_factory = (
            metadata_store_factory if metadata_store_factory else dict
        )
        fields += [
            (
                "metadata",
                type(meta_default_factory()),
                field(default_factory=meta_default_factory),
            )
        ]

    return make_dataclass(
        "DialogueMetricsTracker",
        fields,
        bases=(MetricsTracker,),
        namespace={"metadata_to_agg": metadata_to_agg},
    )


def get_dataset_as_dict(
    file_path_patterns: Union[list[str], str], exclude_file: Optional[str] = None
) -> dict[str, dict]:
    """Read the SGD json dialog data as dictionary with dialog ID as keys.

    Parameters
    ---------
    file_path_patterns
        A list of files to load of a string patterns that can be passed to ``glob.glob`` to retrieve the filenames of
        each dialogue batch in the split to be evaluated.
    exclude_file
        Used to exclude metrics file when loading the data.
    Returns
    -------
    dataset_dict
        A dictionary is where each field is a dialogue ID (as specified in the training data ``dialogue_id`` field)
        and value is a dictionary containing the dialogue in SGD format.
    """
    dataset_dict = {}
    if isinstance(file_path_patterns, list):
        list_fp = file_path_patterns
    else:
        list_fp = sorted(glob.glob(file_path_patterns))
    for fp in list_fp:
        if exclude_file and exclude_file in fp:
            continue
        logging.info("Loading file: %s", fp)
        with open(fp) as f:
            data = json.load(f)
            if isinstance(data, list):
                for dial in data:
                    dataset_dict[dial["dialogue_id"]] = dial
            elif isinstance(data, dict):
                dataset_dict.update(data)
    return dataset_dict


def value_generated(
    value: str,
    utterance: str,
    use_fuzzy_matching: bool = False,
    match_threshold: float = 0.7,
) -> bool:
    """Checks if a given value is present in a generated utterance.

    Parameters
    ----------
    value:
        The value that should be generated.
    utterance:
        A generated utterance that should contain value.
    use_fuzzy_matching:
        If True, a fuzzy matching based on the ``fuzzywuzzy.fuzz.token_sort_ratio`` function is used to check for
        approximate matches. This should help with minor spelling differences and word order reversals.
    match_threshold
        The similarity threshold for determining if a value is in the string.

    Returns
    -------
    A boolean to indicate whether the value is present in the string or not in the input utterance.

    Notes
    -----
    1. There a pitfalls with using fuzzy matching. For example:

        >>> fuzz.token_sort_ratio("San Francisco", "San Fran")/100
        0.76
        >>> fuzz.token_sort_ratio("4:45", "4:45 pm")/100
        0.73
        >>> fuzz.token_sort_ratio("Hakka Restaurant", "restaurant hakka")/100
        1.0
        >>> fuzz.token_sort_ratio("2:00","$2,000")/100
        0.89
        >>> fuzz.token_sort_ratio("2:00pm","$2,000")/100
        0.73

    2. The fuzzy matching works by splitting values by " " and matching against all n-grams of the same order.
    """

    best_match = -float("inf")

    if use_fuzzy_matching:
        n_gram_ord = len(value.split(" "))
        utt_ngrams = ngrams(utterance.split(" "), n_gram_ord)
        for gram in utt_ngrams:
            actual_val = " ".join(gram)
            sim = fuzz.token_sort_ratio(actual_val, value) / 100
            if sim > best_match:
                best_match = sim
                # stop early if a similar enough string is found
                if sim > match_threshold:
                    return True
        return best_match > match_threshold
    else:
        return value in utterance


def compute_f1(list_ref, list_hyp):
    """Compute F1 score from reference (ground truth) list and hypothesis list.
    Args:
      list_ref: List of true elements.
      list_hyp: List of positive (retrieved) elements.
    Returns:
      A F1Scores object containing F1, precision, and recall scores.
    """

    ref = Counter(list_ref)
    hyp = Counter(list_hyp)
    true = sum(ref.values())
    positive = sum(hyp.values())
    true_positive = sum((ref & hyp).values())
    precision = float(true_positive) / positive if positive else 1.0
    recall = float(true_positive) / true if true else 1.0
    if precision + recall > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:  # The F1-score is defined to be 0 if both precision and recall are 0.
        f1 = 0.0

    return F1Scores(f1=f1, precision=precision, recall=recall)


def load_canonical_map(canonical_map_path: str) -> dict:
    try:
        with open(canonical_map_path, "r") as f:
            map = json.load(f)
    except FileNotFoundError:
        logging.warning(
            f"Could not find canonical value map at path: {canonical_map_path}"
        )
        return {}
    return map


def fuzzy_string_match(str_ref, str_hyp):
    """Returns fuzzy string similarity score in range [0.0, 1.0]."""

    # The higher the score, the higher the similarity between the two strings.
    return fuzz.token_sort_ratio(str_ref, str_hyp) / 100.0


def store_f1(
    metrics: list[str],
    tracker: MetricsTracker,
    f1_scores: F1Scores,
    metric_key: str,
    dial_id: str,
):
    """Updates the metrics tracker with F1 scores."""

    if tracker is None:
        return

    for metric in metrics:
        m_store = getattr(tracker, metric)
        metric_name = metric.split("_")[-1]
        assert metric_name in _METRIC_NAMES
        m_store[metric_key].append(getattr(f1_scores, metric_name))

    metadata = getattr(tracker, "metadata")
    # metric_name_key = "igcd_f1" if "inform" in metrics[0].lower() else "rgcd_f1"
    metric_name_key = metrics[0].lower()
    assert dial_id not in metadata[metric_name_key]["dial_id"][metric_key]
    metadata[metric_name_key]["dial_id"][metric_key].append(dial_id)

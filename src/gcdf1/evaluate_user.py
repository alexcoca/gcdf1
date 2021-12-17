from __future__ import annotations

import collections
import json
import os
from collections import defaultdict

from absl import app, flags, logging
from omegaconf import DictConfig, OmegaConf

from gcdf1.metrics import NAN_VAL, USER_METRICS, compute_igcdf1, compute_rgcdf1
from gcdf1.utils.evaluator import (
    MetricsTracker,
    asdict,
    get_dataset_as_dict,
    get_metrics_tracker,
    load_canonical_map,
    metadata_store_factory,
)
from gcdf1.utils.utils import get_commit_hash, get_datetime

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "prediction_dir",
    None,
    "Directory in which all JSON files combined are predictions of the"
    " evaluation set on a single model checkpoint. We evaluate these JSON files"
    " by DSTC8 metrics.",
)
flags.DEFINE_string(
    "output_user_metric_file",
    None,
    "Single JSON output file containing aggregated evaluation metrics results"
    " for all predictions files in FLAGS.prediction_dir. Aggregation is performed across"
    " joint values at intent, service and dialogue level.",
    short_name="o",
)
flags.DEFINE_string(
    "config",
    None,
    "Path to a yaml file containing the configuration for the user evaluator.",
    required=True,
    short_name="c",
)
flags.DEFINE_string(
    "canonical_map_path",
    "",
    "Path containing a mapping from slot to canonical values to other surface forms. Improves Inform F1 robustness.",
    short_name="cmap",
)


PER_DIAL_OUTPUT_FILENAME = "metrics_per_dialogue.json"


def update_hyp_dialogue(
    metrics_tracker: MetricsTracker, dialogue: dict, per_dial_metrics: dict
):
    """Updates the dialogue `metrics` field with additional metrics information."""
    dialogue_id = dialogue["dialogue_id"]
    per_dial_metrics[dialogue_id] = collections.defaultdict(dict)
    tracked_metrics = metrics_tracker.tracked_metrics()
    for metric in tracked_metrics:
        this_metric_values = getattr(metrics_tracker, metric)
        per_dial_metrics[dialogue_id][metric] = this_metric_values
        dialogue["metrics"][metric] = this_metric_values  # noqa
    untracked = [
        metric for metric in metrics_tracker.metrics() if metric not in tracked_metrics
    ]
    for metric in untracked:
        dialogue["metrics"][metric] = NAN_VAL
        per_dial_metrics[dialogue_id][metric] = NAN_VAL


def consistency_checks(hyp_slots: set, ref_slots: set):
    # check if the set of slots in requested user acts is the same
    # as what the user requests
    if hyp_slots != ref_slots:
        logging.warning(
            "Hypothesized request slots set does not exactly match the reference slot set according to goals!\n"
            f"Slots hypothesised but not in goal: {hyp_slots - ref_slots}\n"
            f"Slots in reference but not in hypothesis: {ref_slots- hyp_slots}\n"
        )


def f1_metrics_store_factory() -> defaultdict:
    """Dictionary that stores a domain-level value of a given metric (e.g., F1)."""
    return defaultdict(list)


def get_reference_free_metrics(dataset_hyp: dict, config: DictConfig):
    """Compute metrics when the user model interacts freely with another agent."""

    tracked_user_metrics = USER_METRICS
    canonical_map = None
    if config.igcd_f1.value_matching.use_fuzzy_match:
        if config.igcd_f1.value_matching.use_canonical_map:
            canonical_map = load_canonical_map(
                config.igcd_f1.value_matching.canonical_map_path
            )
        else:
            canonical_map = {}

    user_metrics_store_factory = {}
    for metric in tracked_user_metrics:
        user_metrics_store_factory[metric] = f1_metrics_store_factory

    average_metrics = get_metrics_tracker(
        tracked_user_metrics,
        metrics_store_factory=user_metrics_store_factory,
        metadata=True,
        metadata_store_factory=metadata_store_factory,
    )()
    per_dial_metrics = {}
    goal_req_slots, act_req_slots = set(), set()
    goal_info_slot_vals, act_info_slot_vals = set(), set()
    # track value matching and matching failures by domain in each dialogue

    # calculate metrics at dialogue level
    for dial_id, dial_hyp in dataset_hyp.items():
        logging.info(f"Evaluating dialogue {dial_id}")
        metrics_store = get_metrics_tracker(
            tracked_user_metrics,
            metrics_store_factory=user_metrics_store_factory,
            metadata=True,
            metadata_store_factory=metadata_store_factory,
        )()
        igcd_f1_scores = compute_igcdf1(
            dial_hyp,
            dial_hyp["goal"],
            tracker=metrics_store,
            config=config.igcd_f1,
            canonical_map=canonical_map,
        )  # noqa: F481
        req_f1_scores = compute_rgcdf1(
            dial_hyp,
            dial_hyp["goal"],
            metrics_store,
            config=config.rgcd_f1,
        )
        average_metrics += metrics_store
        goal_req_slots.update(req_f1_scores["metadata"]["rgcd_f1"]["true_pos"])
        act_req_slots.update(req_f1_scores["metadata"]["rgcd_f1"]["pos"])
        goal_info_slot_vals.update(igcd_f1_scores["metadata"]["igcd_f1"]["true_pos"])
        act_info_slot_vals.update(igcd_f1_scores["metadata"]["igcd_f1"]["pos"])
        update_hyp_dialogue(metrics_store, dial_hyp, per_dial_metrics)
    logging.info("Evauation complete, aggregating metrics ...")
    average_metrics.aggregate_metrics(agg_fcn="mean")
    all_dial_metric_aggregated = asdict(average_metrics)
    consistency_checks(act_req_slots, goal_req_slots)

    return (
        all_dial_metric_aggregated,
        per_dial_metrics,
    )


def add_metadata(aggregated_metrics_dict: dict, config: DictConfig):
    """Add evaluation configuration to aggregated metrics file."""

    aggregated_metrics_dict["metadata"]["evaluation_config"] = OmegaConf.to_container(
        config
    )
    commit_hash = get_commit_hash()
    if "fatal" not in commit_hash:
        aggregated_metrics_dict["metadata"]["commit_hash"] = get_commit_hash()
    aggregated_metrics_dict["metadata"]["date"] = get_datetime()


def _main(argv=None):

    config = OmegaConf.load(FLAGS.config)
    use_lowercase = config.use_lowercase

    # update config with relevant flags and casing behaviour
    config.igcd_f1.value_matching.canonical_map_path = FLAGS.canonical_map_path
    config.igcd_f1.use_lowercase = use_lowercase
    config.igcd_f1.value_matching.use_lowercase = use_lowercase
    config.igcd_f1.repetitions.sys_behaviours.use_lowercase = use_lowercase
    config.igcd_f1.repetitions.user_behaviours.use_lowercase = use_lowercase
    config.igcd_f1.constraints_not_in_goal.use_lowercase = (use_lowercase,)
    config.igcd_f1.constraints_not_in_goal.sys_checks.use_lowercase = use_lowercase
    config.rgcd_f1.use_lowercase = use_lowercase
    config.rgcd_f1.repetitions.system_behaviour.use_lowercase = use_lowercase

    dataset_hyp = get_dataset_as_dict(
        os.path.join(FLAGS.prediction_dir, "dialogues_*.json"),
        exclude_file="dialogues_and_metrics.json",
    )

    # evaluate the user against its own goals
    (
        all_dial_agg_metrics,
        per_dial_metrics,
    ) = get_reference_free_metrics(dataset_hyp, config=config)

    # add metadata to the aggregated metric files for reproducibility
    add_metadata(all_dial_agg_metrics, config)
    logging.info("Outputting aggregated metrics file")
    with open(
        os.path.join(FLAGS.prediction_dir, FLAGS.output_user_metric_file), "w"
    ) as f:
        json.dump(
            all_dial_agg_metrics, f, indent=2, separators=(",", ": "), sort_keys=True
        )

    logging.info("Saving dialogues with metrics information ...")
    with open(os.path.join(FLAGS.prediction_dir, PER_DIAL_OUTPUT_FILENAME), "w") as f:
        json.dump(dataset_hyp, f, indent=2, separators=(",", ": "), sort_keys=True)


def main():
    flags.mark_flag_as_required("prediction_dir")
    flags.mark_flag_as_required("output_user_metric_file")
    app.run(_main)


if __name__ == "__main__":
    flags.mark_flag_as_required("prediction_dir")
    flags.mark_flag_as_required("output_user_metric_file")
    app.run(_main)

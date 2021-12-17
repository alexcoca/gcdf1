"""This module contains helpers for analysing the evaluator output."""
from __future__ import annotations

from collections import defaultdict
from typing import Literal, Optional

from omegaconf import OmegaConf

from gcdf1.metrics import get_domain_requests
from gcdf1.multiwoz_metadata import (
    DOMAIN_CONSTRAINT_GOAL_KEYS,
    INFORM_ACT_PATTERNS,
    REQUEST_ACT_PATTERNS,
)
from gcdf1.utils.data import split_iterator
from gcdf1.utils.dialogue import SLOT_VALUE_SEP
from gcdf1.utils.metrics import get_parameters_by_service
from gcdf1.utils.utils import (
    cast_vals_to_set,
    default_to_regular,
    nested_defaultdict,
    store_data,
)


def analyse_simple_context_feature(dialogues: dict, keys: list[str]):
    """This function should be used as a tool to analyse "simple" context-driven
    features which store only constraint info broken down by dialogue ID::

        {
        'context_driven_feature_name': {'dial_id': ['constraint_info', ...]}
        }

    Contrast this with "hierarchical" context-driven features which have intermediate keys detailing specific
    user or system behaviours::

        {
            'context_driven_feature_name': { 'behaviour_name': {'dial_id': {'domain': [constraint_info, ...], ....}}}
        }

    Parameters
    ----------
    dialogues
        This is the evaluator output for a simple context-driven feature, which one of the keys under
        ['metadata']['metric_name'] in the evaluator output .json file. Here metric is 'igcd_f1` or `rgcd_f1`
        and the format is as follows::

            {
                'dialogue_id': { 'domain_name': [constraint_info, ...]}
            }

        where `constraint_info` is a string formatted as 'slot{SLOT_VALUE_SEP}value'. Import
        SLOT_VALUE_SEP from ``gcdf1.utils.dialogues``.
    keys
        Controls the content of the output mapping. Possible values are ``['domain', 'dialogue']``
        and ``['domain', 'slot', 'dialogue']``. See ``analysis_output`` docs to understand
        their effect.

    Returns
    -------
    analysis_output
        The structure of the output is:

            - ``['domain', 'dialogue']``: in this case, the output mapping is of the form::

                    {
                        'domain_name': ['dialogue_id', ...]
                    }

                which allows you to obtain statistics for the different context driven features
                and to query the generated conversations to analyse your model output systematically.

            - ``['domain', 'slot', 'dialogue']``: in this case, the output mapping is of the form::

                {
                    'domain_name': {'slot_name': ['dialogue_id', ...]}

                }

            which allows a fine-grained slot-level output analysis.
    """
    analysis_output = nested_defaultdict(list, depth=len(keys) - 1)  # type: defaultdict

    for dial_id in dialogues:
        context_feat_dial = dialogues[dial_id]
        for domain in context_feat_dial:
            if keys == ["domain", "dialogue"]:
                store_key = [domain]
                info_to_store = dial_id
                store_data(
                    info_to_store,
                    analysis_output,
                    store_key,
                )
                break
            elif keys == ["domain", "slot", "dialogue"]:
                info_to_store = dial_id
                for constraint in context_feat_dial[domain]:
                    slot = constraint.split(SLOT_VALUE_SEP)[0]
                    store_key = [domain, slot]
                    store_data(info_to_store, analysis_output, store_key)
            else:
                raise ValueError("Undefined key combination!")

    return default_to_regular(cast_vals_to_set(analysis_output))


def analyse_hierarchical_context_feature(
    dialogue_hierarchy: dict,
    keys: list[str],
    new_behaviour_names: Optional[dict[str, str]] = None,
    store_dial_id: bool = False,
):
    """This function should be used as a tool to analyse "hierarchical" context-driven
    features which which have intermediate keys detailing specific user or system behaviours::

        {
            'context_driven_feature_name': { 'behaviour_name': {'dial_id': {'domain': [constraint_info', ...], ....}}}
        }

    Contrast this with "simple" context-driven features::

        {
            'context_driven_feature_name': {'dial_id': ['constraint_info', ...]}
        }

    Parameters
    ----------
    dialogue_hierarchy
         This is the evaluator output for a hierarchical context-driven feature, which one of the keys under
        ['metadata']['metric_name'] in the evaluator output .json file. Here metric is 'igcd_f1` or `rgcd_f1`
        and the format is as follows::

            {
                'dialogue_id': { 'domain_name': [constraint_info, ...]}
            }

        where `constraint_info` is a string formatted as 'slot{SLOT_VALUE_SEP}value'. Import
        SLOT_VALUE_SEP from ``gcdf1.utils.dialogues``.
    keys
        Controls the content of the output mapping. Possible values are ``['match_key' ,'domain']``
        and ``['match_key', 'domain', 'slot']``. See ``analysis_output`` docs to understand
        their effect.
    new_behaviour_names
        This mapping ({'evaluator_bevaviour_names': 'my_behaviour_name', ...}) can be used to change the names
        of the behaviours in the output analysis. This can combine statistics for different behaviours (set same
        value for multiple keys in the evaluator output) or to simply rename the behaviours.
    store_dial_id
        If `True` the output mapping will contain a list of dialogue IDs at the lowest level instead of
        constraint information.


    Returns
    -------
    analysis_output
        A mapping containing analysis output. If `keys` is set to:

         - ``['match_key' ,'domain']`` then the output mapping has the form::

            {
            'behaviour_name': {'domain': ['dial_id', ...]}, ...
            }

        which facilitates analysis of behaviour statistics and retrieval of conversations which
        match certain behaviour patterns.

        -  ['match_key', 'domain', 'slot'] then the structure of the output is::

            {
            'behaviour_name': {'domain': {'slot': ['info', ....]}}, ...
            }

        where `info` is the dialogue_id if `store_dial_id=True` and is 'slot{SLOT_VALUE_SEP}value' otherwise. Import
        SLOT_VALUE_SEP from ``gcdf1.utils.dialogues``
    """

    if not new_behaviour_names:
        new_behaviour_names = {}

    analysis_output = nested_defaultdict(list, depth=len(keys))

    for orig_behaviour_name in dialogue_hierarchy:
        out_behaviour_name = new_behaviour_names.get(
            orig_behaviour_name, orig_behaviour_name
        )
        this_reason_matches = dialogue_hierarchy[orig_behaviour_name]
        for dial_id in this_reason_matches:
            for domain in this_reason_matches[dial_id]:
                if keys == ["match_key", "domain"]:
                    store_key = [out_behaviour_name, domain]
                    info_to_store = dial_id
                    store_data(
                        info_to_store,
                        analysis_output,
                        store_key,
                    )
                elif keys == ["match_key", "domain", "slot"]:
                    store_key = [out_behaviour_name, domain]
                    for matched_constr_dict in this_reason_matches[dial_id][domain]:
                        for constr in matched_constr_dict:
                            slot = constr.split(SLOT_VALUE_SEP)[0]
                            info_to_store = constr
                            if store_dial_id:
                                info_to_store = dial_id
                            store_data(
                                info_to_store, analysis_output, store_key + [slot]
                            )

                else:
                    raise ValueError

    return default_to_regular(analysis_output)


def parse_requests(dial, config_path, agent: str = "SYSTEM"):

    assert agent in ["USER", "SYSTEM"]
    user_turns, system_turns = False, True
    if agent == "USER":
        user_turns, system_turns = True, False

    config = OmegaConf.load(config_path)
    config.igcd_f1.use_lowercase = True
    config.igcd_f1.value_matching.use_fuzzy_match = False
    requests = get_parameters_by_service(
        dial,
        dial["goal"],
        config.igcd_f1,
        user_turns=user_turns,
        system_turns=system_turns,
        act_patterns=REQUEST_ACT_PATTERNS,
    )
    return requests


def analyse_req_expression_patterns(
    split: Literal["test"],
    domains: list[str],
    data_path: str,
    config_path: str,
    split_by_dial_type: bool = False,
):
    """Find out how often the user requests all information in a single turn vs across multiple turns.
    Dialogues where only one slot is to be requested are excluded from analysis.

    Parameters
    ----------
    split:
        The split from which dialogues are to be retrieved.
    domains:
        The domains for which information request patterns are to be analysed.
    data_path:
        Path to the generated conversations.
    config_path:
        Path to the evaluator configuration.
    split_by_dial_type:
        If `True` the output mapping will contain an additional key which can be
        `sng` or `mul` to indicate whether the dialogue is single- or multi-domain

    Returns
    -------
    reqs_expression_counter
        A mapping with structure::

            {
            'domain_name': { 'expression_pattern_key': [dial_id, ...]}, ...
            }

        if `split_by_dial_type=False` and::

            {
                'domain_name': {
                    'expression_pattern_key': {
                        'sng':  [dial_id, ...],
                        'mul':  [dial_id, ...],
                    }
                }, ...
            }

        otherwise. `expression_pattern_key` can be:

            - ``'miss_all_req'``: no information was requested by the user

            - ``{miss|no_miss}_{one|mult}``: the four values are determined as follows:

                    * ``{miss|no_miss}``: ``no_miss`` if the user requested all info in the goal, ``miss`` otherwise

                    * ``{one|mult}``: ``one`` if user requested all info in one turn, ``mult`` otherwise.
    """

    depth = 3 if split_by_dial_type else 2
    reqs_expression_counter = nested_defaultdict(list, depth=depth)

    for _, dial in split_iterator(
        split,
        data_pckg_or_path=data_path,
    ):
        dial_id = dial["dialogue_id"]
        dialogue_type = "sng" if dial_id.startswith("SNG") else "mul"
        for domain in domains:
            goal_reqs = get_domain_requests(dial["goal"], domain, use_lowercase=True)
            goal_reqs = {f"{req}{SLOT_VALUE_SEP}?" for req in goal_reqs}
            # skip dials where there is one slot to be requested
            if len(goal_reqs) == 1 or not goal_reqs:
                continue
            usr_reqs = parse_requests(dial, config_path, agent="USER")[domain]
            # indices of turns where the user requested the information
            turns_expressed = [set(usr_reqs.get(req, set())) for req in goal_reqs]
            # check if user failed to request info
            missed_req_idx = []
            for i, turns_idxs in enumerate(turns_expressed):
                if not turns_idxs:
                    missed_req_idx.append(i)
            # IDs of the turns where the info was requested
            turns_expressed = [
                turns_expressed[i]
                for i in range(len(turns_expressed))
                if i not in missed_req_idx
            ]
            # no info requested
            if not turns_expressed:
                if split_by_dial_type:
                    reqs_expression_counter[domain][dialogue_type][
                        "miss_all_reqs"
                    ].append(dial_id)
                else:
                    reqs_expression_counter[domain]["miss_all_reqs"].append(dial_id)
                continue
            # request made across multiple turns if the intersection of the turn_id sets is non-empty
            common_turns = turns_expressed[0].intersection(*turns_expressed)
            second_key = "miss" if missed_req_idx else "no_miss"
            first_key = "one" if common_turns else "mult"
            store_key = f"{second_key}_{first_key}"
            # add single/multi-domain key to output dict
            if split_by_dial_type:
                reqs_expression_counter[domain][dialogue_type][store_key].append(
                    dial_id
                )
            else:
                reqs_expression_counter[domain][store_key].append(dial_id)

    return default_to_regular(reqs_expression_counter)


def analyse_value_matching_output(proc_constraints: dict, keys: list[str]):
    """The evaluator attempts to convert constraint values to canonical forms before
    computed I-GCDF1 scores. This function helps analyse the output of the value pre-processing
    feature.

    Parameters
    ----------
    proc_constraints
        Output of the evaluator value pre-processing algorithm. This is stored in the following fields:

            - ['metadata']['igcd_f1']['preprocessed_sys_constraints']
            - ['metadata']['igcd_f1']['preprocessed_sys_nlu']
            - ['metadata']['igcd_f1']['preprocessed_user_constraints']

        These fields contain nested dictionaries of the form::

            {
                'dialogue_id':{
                    'match_outcome': {
                        'domain_name':{'slot_name': [match_algo_output_dict, ...]}
                        }
                }
            }

            Here `match_outcome` can be `values_matched` or `matching_failures`. A sample `match_algo_output_dict`::

                {
                    "best_match": null,
                    "canonical_value": null,
                    "hyp_value": "same area",
                    "method": "extended"
                },

            `method` indicates if the match is against one of the goal values ("simple") or against the entire canonical
             map ("extended"). The `canonical_value` and `best_match` field are populated if a match is found.
            `best_match` represents the value in the canonical map that matches the value extracted from the
            conversation (`hyp_value`). For example::

                {
                    "best_match": "meze bar",
                    "canonical_value": "meze bar restaurant",
                    "hyp_value": "meze bar",
                    "method": "extended"
                }

    keys
        A list of keys that determine the structure of the output. Possible values are:

            - ['match_type', 'domain', 'slot']
            - ['match_type', 'domain']
            - ['match_type', 'slot']

    Returns
    -------
    analysis_output
        The innermost value of this nested mapping (`match_res_tuple`) is a (hyp_value, canonical_value) tuple,
        where the first entry is the value that is not in canonical form and the second is the canonical value
        corresponding to it (None if hyp_value was not sufficiently similar to a canonical map value). The outermost
        keys are:

            - {'match_outcome': {'domain_name': {'slot_name': match_res_tuple}}} if keys=['match_type', 'domain', 'slot']

            - {'match_outcome': {'domain_name': match_res_tuple} if keys=['match_type', 'slot']

            - {'match_outcome': {'slot_name': match_res_tuple} if keys=['match_type', 'domain']
    """  # noqa
    analysis_output = nested_defaultdict(list, depth=len(keys))

    for dial_id in proc_constraints:
        dial_matches = proc_constraints[dial_id]
        for match_key in dial_matches:
            this_key_matches = dial_matches[match_key]
            for domain in this_key_matches:
                for matched_slot in this_key_matches[domain]:
                    if keys == ["match_type", "domain", "slot"]:
                        store_key = [match_key, domain, matched_slot]
                    elif keys == ["match_type", "domain"]:
                        store_key = [match_key, domain]
                    elif keys == ["match_type", "slot"]:
                        store_key = [match_key, matched_slot]
                    else:
                        raise ValueError
                    matched_slot_data = this_key_matches[domain][
                        matched_slot
                    ]  # type: list[dict]
                    store_data(
                        (
                            matched_slot_data[-1]["hyp_value"],
                            matched_slot_data[-1]["canonical_value"],
                        ),
                        analysis_output,
                        store_key,
                    )
    return analysis_output


def parse_constraints(
    dialogue, config_path: str, canonical_map: dict = None
) -> dict[str, dict[str, list[int]]]:
    """This function retrieves the constraints expressed by the user. See
    `get_prarameters_by_service` for output structure."""

    config = OmegaConf.load(config_path)
    config.igcd_f1.use_lowercase = True
    config.igcd_f1.value_matching.use_lowercase = config.igcd_f1.use_lowercase
    constraints = get_parameters_by_service(
        dialogue,
        dialogue["goal"],
        config.igcd_f1,
        system_turns=False,
        act_patterns=INFORM_ACT_PATTERNS,
        canonical_map=canonical_map,
    )

    return constraints


def get_goal_constraints_by_type(
    goal: dict, use_lowercase: bool = True
) -> dict[str, dict[str, list[str]]]:
    """Extract constraints for all domains from the dialogue goal.

    Parameters
    ----------
    goal
    use_lowercase
        If `True` slots and values are lowercased.
    Returns
    -------
    goal_constraints_by_service
        A mapping of the form::

            {
                'domain': {'goal_type': [constraint, ...]},
            }

        Each constraint is formatted as "{slot}{SLOT_VALUE_SEP}{value}". See gcdf1.utils.dialogue for
        SLOT_VALUE_SEP. `goal_type` is defined in `gcdf1.multiwoz_metadata.`

    Notes:
    -----
        The extraction proceeds by extracting all the slots and values from the
        keys specified in `gcdf1.multiwoz_metadata.DOMAIN_CONSTRAINT_GOAL_KEYS` and the keys
        of the output contain duplicates.
    """

    goal_constraints = defaultdict(lambda: defaultdict(list))

    for domain in goal:
        # extract information from domain sub-goals (except reqt)
        # this will contain duplicates depending on fail_* subgoals content
        for subgoal_key in DOMAIN_CONSTRAINT_GOAL_KEYS:
            subgoal_slot_value_map = goal[domain].get(subgoal_key, {})
            if subgoal_slot_value_map:
                for name, value in subgoal_slot_value_map.items():
                    if use_lowercase:
                        name, value = name.lower(), value.lower()
                    goal_constraints[domain][subgoal_key].append(
                        f"{name}{SLOT_VALUE_SEP}{value}"
                    )

    return dict(goal_constraints)


def analyse_info_expression_patterns(
    split: str,
    domains: str,
    data_path: str,
    config_path: str,
    keys_checked: list[str],
    split_by_dial_type: bool = False,
    merge_subgoal_key: bool = True,
    canonical_map: Optional[dict] = None,
):
    """
    Find out how often the user provides all constraints in a single turn vs across multiple turns.
    Dialogues where only one piece of information is to be provided are excluded from analysis.

    Parameters
    ----------
    split:
        The split from which dialogues are to be retrieved.
    domains:
        The domains for which information request patterns are to be analysed.
    data_path:
        Path to the generated conversations.
    config_path:
        Path to the evaluator configuration.
    keys_checked:
        Contains any non-empty subset of ['info', 'book', 'fail_info', 'fail_book'], for which the
        constraint expression pattern is to be checked.
    split_by_dial_type:
        If `True` the output mapping will contain an additional key which can be
        `sng` or `mul` to indicate whether the dialogue is single- or multi-domain
    merge_subgoal_key
        MultiWOZ 2.1 goals contain `info`, `book` keys to differentiate between search and booking constraints.
        The fields `fail_info` and `fail_book` store the constraints to be expressed if there are no search
        results or the booking does not succeed. If this parameter is set to `True` then the output dictionary
        contains only `search` and `booking` categories - the data from `info`/`fail_info` and `book`/`fail_book` is
        merged.
    canonical_map
        This is a dictionary containing paraphrases for each slot value, used to improve F1 computation robustness by
        mapping the values to their canonical forms prior to F1 computation.

    Returns
    -------
    constraints_expression_counter
        A mapping with structure::

            {
            'domain_name': { 'constraint_type': {'expression_pattern_key': [dial_id, ...]}, ...
            }

        if `split_by_dial_type=False` and::

            {
                'domain_name': {
                    `constraint_type`: {
                        'expression_pattern_key': {
                            'sng':  [dial_id, ...],
                            'mul':  [dial_id, ...],
                        }
                    }
                }, ...
            }

        otherwise. `constraint_type` is `search` or `booking` if `merge_subgoal_key=True` and can be any value in
        `keys_checked` otherwise.  `expression_pattern_key` can be:

            - ``'miss_all_req'``: no information was provided by the user

            - ``{miss|no_miss}_{one|mult}``: the four values are determined as follows:

                    * ``{miss|no_miss}``: ``no_miss`` if the user provided all info in the goal, ``miss`` otherwise

                    * ``{one|mult}``: ``one`` if user provided all info in one turn, ``mult`` otherwise.
    """

    depth = 4 if split_by_dial_type else 3
    constraints_expression_counter = nested_defaultdict(list, depth=depth)

    for _, dial in split_iterator(
        split,
        data_pckg_or_path=data_path,
    ):
        dial_id = dial["dialogue_id"]
        dialogue_type = "sng" if dial_id.startswith("SNG") else "mul"
        for domain in domains:
            goal_constr = get_goal_constraints_by_type(dial["goal"])
            if domain not in goal_constr:
                continue
            usr_constraints = parse_constraints(
                dial, config_path, canonical_map=canonical_map
            )[domain]
            # if not usr_constraints:
            #     continue
            for key in keys_checked:
                if key not in dial["goal"][domain]:
                    continue
                if merge_subgoal_key:
                    constr_type = "search" if "info" in key else "booking"
                else:
                    constr_type = key
                if len(goal_constr[domain][key]) == 1:
                    continue
                # in which turn a constraint has been expressed?
                turns_expressed = [
                    set(usr_constraints.get(constr, set()))
                    for constr in goal_constr[domain][key]
                ]
                missed_constr_idx = []
                for i, turns_idxs in enumerate(turns_expressed):
                    if not turns_idxs:
                        missed_constr_idx.append(i)
                turns_expressed = [
                    turns_expressed[i]
                    for i in range(len(turns_expressed))
                    if i not in missed_constr_idx
                ]
                if not turns_expressed:

                    if split_by_dial_type:
                        constraints_expression_counter[dialogue_type][constr_type][
                            "miss_all"
                        ][domain].append(dial_id)
                    else:
                        constraints_expression_counter[constr_type]["miss_all"][
                            domain
                        ].append(dial_id)
                    continue

                common_turns = turns_expressed[0].intersection(*turns_expressed)
                second_key = "miss" if missed_constr_idx else "no_miss"
                first_key = "one" if common_turns else "mult"
                store_key = f"{second_key}_{first_key}"
                if split_by_dial_type:
                    constraints_expression_counter[dialogue_type][constr_type][
                        store_key
                    ][domain].append(dial_id)
                else:
                    constraints_expression_counter[constr_type][store_key][
                        domain
                    ].append(dial_id)
    return default_to_regular(constraints_expression_counter)

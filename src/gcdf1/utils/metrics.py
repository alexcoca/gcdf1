from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from contextlib import suppress
from itertools import chain
from typing import Optional, OrderedDict, Union

from omegaconf import DictConfig

from gcdf1.multiwoz_metadata import (
    COREFERENCE_VALUES,
    DOMAIN_CONSTRAINT_GOAL_KEYS,
    INFORM_ACT_PATTERNS,
    MULTIPLE_VALUE_PROCESSING_EXCEPTIONS,
    REQUEST_ACT_PATTERNS,
)
from gcdf1.utils.data import dialogue_iterator
from gcdf1.utils.dialogue import (
    ACT_SLOT_SEP,
    MULTIPLE_VALUE_SEP,
    SLOT_VALUE_SEP,
    SYSTEM_AGENT,
    USER_AGENT,
    get_turn_actions,
    get_turn_idx,
)
from gcdf1.utils.evaluator import MetricsTracker, fuzzy_string_match
from gcdf1.utils.utils import append_to_values, cast_vals_to_set, safeget


def process_multiple_values(actions: dict[str, list[str]], merge_values: bool):
    """Preprocesses actions where the ``value`` field contains more than one element. Input
    actions modified in place.

    Parameters
    ----------
    actions
        A mapping from service names to actions in that service.
    merge_values
        If True, the separator MULTIPLE_VALUE_SEP is replaced by " " in each action.
        Otherwise, an action is created for each component of the value.

    Notes
    -----
    Use the MULTIPLE_VALUE_PROCESSING_EXCEPTIONS in multiwoz_metadata.py to prevent multiple actions
    from being created for the same slot when `merge_value=False` for MultiWOZ 2.1.
    """

    # TODO: TEST THIS FUNCTION

    for service in actions:
        this_serv_actions = actions[service]
        i_params_to_proc = [
            idx
            for idx, param in enumerate(this_serv_actions)
            if MULTIPLE_VALUE_SEP in param
        ]
        i_params_to_proc = reversed(sorted(i_params_to_proc))
        # handle multiple values for one slot in the same turn
        if merge_values:
            for idx in i_params_to_proc:
                this_serv_actions[idx] = this_serv_actions[idx].replace(
                    MULTIPLE_VALUE_SEP, " "
                )
        else:
            multiple_val_actions = [this_serv_actions.pop(i) for i in i_params_to_proc]
            single_val_actions = []  # type: list[str]
            for m_val_action in multiple_val_actions:
                slot, *values = m_val_action.split(SLOT_VALUE_SEP)
                # merge values back in case there was a SLOT_VALUE_SEP symbol in value
                value = f"{SLOT_VALUE_SEP}".join(values)
                # avoid splitting slots such address, phone number or postcode
                if slot in MULTIPLE_VALUE_PROCESSING_EXCEPTIONS:
                    single_val_actions.append(
                        f"{slot}{SLOT_VALUE_SEP}{value.replace(MULTIPLE_VALUE_SEP, ' ')}"
                    )
                else:
                    single_val_actions.extend(
                        [
                            f"{slot}{SLOT_VALUE_SEP}{v}"
                            for v in value.split(MULTIPLE_VALUE_SEP)
                        ]
                    )
            for v in single_val_actions:
                assert isinstance(v, str)
            this_serv_actions += single_val_actions


def find_best_value_match(
    match_values: list[str], hyp_constr_value: str, match_threshold: float
) -> Union[str, None]:
    """Find the best match of a value against a list of potential matches.

    Parameters
    ----------
    hyp_constr_value
        Value to be matched.
    match_values
        Strings amongst which the best match is searched.
    match_threshold
        Threshold for Levenstein distance above which a candidate is considered a match.
    """
    best_match_score = -float("inf")
    best_score_idx = 0
    for i, pos_value in enumerate(match_values):
        score = fuzzy_string_match(hyp_constr_value, pos_value)
        if score > best_match_score:
            best_match_score = score
            best_score_idx = i

    if best_match_score > match_threshold or math.isclose(
        match_threshold, best_match_score
    ):
        return match_values[best_score_idx]
    return


def get_domain_goal_slot_values(
    domain_goal: dict, slot: str, use_lowercase: bool = True
) -> set[str]:
    """Retrieve the values of a slot from a domain_goal."""
    slot_vals = set()
    for key in DOMAIN_CONSTRAINT_GOAL_KEYS:
        subgoal_slot_values = domain_goal.get(key, {})
        if slot in subgoal_slot_values:
            assert isinstance(subgoal_slot_values[slot], str)
            val = (
                subgoal_slot_values[slot].lower()
                if use_lowercase
                else subgoal_slot_values[slot]
            )
            slot_vals.add(val)
    return slot_vals


def get_canonical_value(
    canonical_map: dict, domain: str, slot: str, best_match: str
) -> str:
    """Find the canonical value of a slot from a certain slot in a given domain, given its best match."""
    canonical_map = canonical_map[slot]
    # search the canonical map first
    if domain in canonical_map and isinstance(canonical_map[domain], dict):
        canonical_map = canonical_map[domain]
    for c_value in canonical_map:
        if c_value == "special_keys":
            continue
        if best_match in canonical_map[c_value]:
            return c_value
    not_in_goal = canonical_map["special_keys"]["not_in_goal"]
    if domain in not_in_goal:
        not_in_goal = not_in_goal[domain]
    for c_value in not_in_goal:
        if best_match in not_in_goal[c_value]:
            return c_value


def get_canonical_values_not_in_goals(
    slot_cmap: dict[str, dict], domain: str
) -> dict[str, set[str]]:
    """Some canonical values do not appear in the goals so they are in a special field in the
     canonical map. This is specific to MultiWOZ 2.1.

    Parameters
    ----------
    slot_cmap
        Canonical map of a given slot. This contains a `special_keys` field.
    domain
        For some slots, the canonical map is split by domain, so the domain is necessary
        in order to correctly retrieve the cannonical values not in goal.
    """
    # for some slots (e.g., hotel-stay) canonical vals outside goals are
    # split by domain
    new_vals = slot_cmap["special_keys"].get("not_in_goal", {}).get(domain, {})
    if not new_vals:
        new_vals = slot_cmap["special_keys"].get("not_in_goal", {})
    return new_vals


def preprocess_constraint_values(
    hyp_constraints: dict[str, dict[str, list[int]]],
    goal: dict,
    config,
    tracker: Optional[MetricsTracker] = None,
    canonical_map: dict = None,
    metadata_fields: Optional[list[str]] = None,
):
    """Map the values of the constraints to canonical form for the MultiWOZ 2.1 corpus. Modifies input in place.
    The algorithm is based on:

        - a canonical map, mapping slots to canonical values to sets of value paraphrases.
        See resources/multiwoz21_canonical_map.json for a MultiWOZ 2.1 example

        - the values that the user is expected to generate.

    The matching is first against the values in the goal and then against the canonical map. Constaraint values from
    domains that are not in the `goal` are ignored.

    Parameters
    ----------
    hyp_constraints
        A mapping of the form::

            {
                'domain': {'slot{SLOT_VALUE_SEP}value': list[int]}
            }
        containing a mapping of domains to action parameters to the turn indices where the parameters are stated
    goal
        User goal. Preprocessing only performed for domains in goal and _not_ for all. Values in the goal are used
        for fuzzy matching purposes.
    tracker
        Used to store the output of the matching procedure. The results will be under
        [*][dial_id][match_key][domain][slot] where match_key is "values_matched" or "matching_failure" depending on
        whether a canonical form was found for the value or not. [*] = `metadata_fields[:-1]
    config
        Only determinise casing behaviour (i.e., if values are lowercased)
    canonical_map
        Corpus-dependent paraphrase table. See resources/multiwoz/multiwoz21_canonical_map.json for a MultiWOZ 2.1
        example.
    metadata_fields
        Where data should be stored in the evaluator output. Last element must be a dialogue ID.
    """

    def track_matches(tracker: defaultdict, match_results: list[dict]):
        """Logs attempts to map a value to a canonical form."""

        if tracker is None:
            return

        nonlocal dial_id
        matches = [bool(result["best_match"]) for result in match_results]
        if all(matches) or any(matches):
            match_idx = [i for i, m in enumerate(matches) if m][0]
        else:
            assert not all(matches)
            match_idx = -1
        tracked_result = match_results[match_idx]
        key, domain, slot = (
            tracked_result.pop("key"),
            tracked_result.pop("domain"),
            tracked_result.pop("slot"),
        )
        tracker[dial_id][key][domain][slot].append(match_results[match_idx])

    metadata_fields = metadata_fields if metadata_fields else ["igcd_f1"]
    dial_id = metadata_fields[-1]
    tracker_metadata = None
    if hasattr(tracker, "metadata"):
        tracker_metadata = safeget(
            getattr(tracker, "metadata"), *metadata_fields[:-1]
        )  # type: defaultdict
    match_threshold = config.fuzzy_match_threshold
    if canonical_map:
        cast_vals_to_set(canonical_map)
    for domain in goal:
        # user has not spoken about a domain at all
        if domain not in hyp_constraints:
            continue
        # extract slots and values informed for this domain
        this_domain_slots = [
            svp.split(f"{SLOT_VALUE_SEP}")[0] for svp in hyp_constraints[domain]
        ]
        this_domain_slot_value_map = defaultdict(set)
        for slot in this_domain_slots:
            assert ACT_SLOT_SEP not in slot
            for const in hyp_constraints[domain]:
                if slot in const:
                    this_domain_slot_value_map[slot].add(
                        f"{SLOT_VALUE_SEP}".join(const.split(SLOT_VALUE_SEP)[1:])
                    )

        for slot, values in this_domain_slot_value_map.items():
            # if lowercase is not used, then arriveBy/leaveAt will cause a lookup error
            if canonical_map:
                # happens for e.g. system slots
                if not any((slot in canonical_map, slot.lower() in canonical_map)):
                    continue
                else:
                    try:
                        this_slot_canonical_map = canonical_map[slot]
                    except KeyError:
                        this_slot_canonical_map = canonical_map[slot.lower()]
                # check if the value is in canonical form or otherwise
                if domain in this_slot_canonical_map and isinstance(
                    this_slot_canonical_map[domain], dict
                ):
                    # slot cannonical map split by domain for some slots;
                    # second check because hotel is both value/domain
                    this_slot_canonical_map = this_slot_canonical_map[domain]
                known_canonical_vals = set(this_slot_canonical_map.keys())
                known_canonical_vals.update(
                    get_canonical_values_not_in_goals(
                        this_slot_canonical_map, domain=domain
                    ).keys()
                )
            else:
                # if we don't know a full list of canonical values, we consider the values in the goal as canonical
                known_canonical_vals = set(
                    get_domain_goal_slot_values(goal.get(domain, {}), slot)
                )
            non_canonical_vals = [
                val for val in values if val not in known_canonical_vals
            ]
            for i, non_can_value in enumerate(non_canonical_vals):
                # avoid trying to match corefs as they can match
                # arbitrary names (e.g., "the restaurant -> the j restaurant")
                if non_can_value in COREFERENCE_VALUES.get(domain, []):
                    continue
                match_values_dict = OrderedDict()
                if canonical_map:
                    goal_slot_vals = list(
                        get_domain_goal_slot_values(
                            goal.get(domain, {}),
                            slot,
                            use_lowercase=config.use_lowercase,
                        )
                    )
                    # consider only the likely candidates first for matching
                    match_values_dict["simple"] = list(
                        chain(
                            *[
                                list(this_slot_canonical_map[c_val])
                                for c_val in goal_slot_vals
                            ]
                        )
                    )

                    # we don't know what this value is supposed to be so we consider all
                    # the possible values in the canonical map and choose the best match
                    match_values_dict["extended"] = list(
                        chain(
                            *[
                                list(this_slot_canonical_map[c_value])
                                for c_value in this_slot_canonical_map
                                if c_value != "special_keys"
                            ]
                        )
                    )
                    if "special_keys" in match_values_dict["extended"]:
                        logging.warning(
                            f"While attempting to match canonical values, "
                            f"found the following constraints: {hyp_constraints[domain]}. Some slots may be "
                            f"out of domain?"
                        )
                    # consider valid values not in goal for any dialogue in the corpus
                    not_in_goal_vals = get_canonical_values_not_in_goals(
                        this_slot_canonical_map, domain=domain
                    )
                    not_in_goal_matches = list(
                        chain(
                            *[
                                list(not_in_goal_vals[c_value])
                                for c_value in not_in_goal_vals
                            ]
                        )
                    )
                    match_values_dict["extended"] += not_in_goal_matches
                else:
                    # try to match the value against the values in goal
                    match_values_dict["simple"] = list(known_canonical_vals)
                    # the slot is not in goal, so we skip matching this
                    if not match_values_dict["simple"]:
                        continue

                match_results = []
                for match_key, match_values in match_values_dict.items():
                    assert isinstance(match_values, list)
                    # if pre-processing values not in current goal, match_values_dict["simple"] is empty
                    if not match_values:
                        continue
                    best_match = find_best_value_match(
                        match_values, non_can_value, match_threshold
                    )
                    if canonical_map:
                        if best_match:
                            canonical_value = get_canonical_value(
                                canonical_map, domain, slot, best_match
                            )
                            assert canonical_value
                        else:
                            canonical_value = None
                    else:
                        canonical_value = best_match
                    if best_match:
                        turn_idx = hyp_constraints[domain].pop(
                            f"{slot}{SLOT_VALUE_SEP}{non_can_value}"
                        )
                        assert isinstance(turn_idx, list)
                        assert all(isinstance(i, int) for i in turn_idx)
                        canonical_action = f"{slot}{SLOT_VALUE_SEP}{canonical_value}"
                        # canonical action might also be expressed
                        if canonical_action in hyp_constraints[domain]:
                            hyp_constraints[domain][canonical_action] += turn_idx
                            hyp_constraints[domain][canonical_action].sort()
                        else:
                            hyp_constraints[domain][canonical_action] = turn_idx
                    key = "values_matched" if best_match else "matching_failures"
                    match_results.append(
                        {
                            "best_match": best_match,
                            "hyp_value": non_can_value,
                            "canonical_value": canonical_value,
                            "method": match_key,
                            "domain": domain,
                            "slot": slot,
                            "key": key,
                        }
                    )
                    # got the job done searching in the likely value set
                    # don't need to check the entire map
                    if best_match:
                        assert match_results
                        break
                track_matches(tracker_metadata, match_results)


def get_parameters_by_service(
    dialogue: dict,
    goal: dict,
    config,
    tracker: Optional = None,
    user_turns: bool = True,
    system_turns: bool = True,
    act_patterns: Optional[list[str]] = None,
    canonical_map: Optional[dict] = None,
    metadata_fields: Optional[list[str]] = None,
) -> dict[str, dict[str, list[int]]]:
    """Retrieve a mapping of the parameters of actions in a dialogue to the turns where they occur, for all services
    in a dialogue.

    Parameters
    ----------
    dialogue
        Dialogue from which action parameters are to be extracted.
    goal
        Used to map action values to canonical form if canonical map is not available.
    config
        Used to control:

            - behaviour when the same slot has multiple values in the same turn

            - lowercase behaviour

            - value matching feature behaviour
    tracker
        This object is used to log the output of the value conversion to canonical form feature.
    user_turns, system_turns
        These flags can be used to retrieve parameters only for one agent.
    act_patterns
        Allows to retrieve parameters only for given actions.
    canonical_map
        A corpus-specific dictionary used to map values to canonical forms to that the F1 score
        calculation is robust.
    metadata_fields
        A list of strings which is a nested key in the evaluator output. This is where the output
        of the matching to canonical form is stored.

    Returns
    -------
    A nested mapping of the form::

        {
            'service_name': {'action_parameter': [turn_idx, ...]}
        }

        where action parameter is a string (e.g., slot or slot_value pair) formatted
        as '{slot}{SLOT_VALUE_SEP}{value}' with separators defined in gcdf1.utils.dialogue.
        The innermost value is a list of integers containing the turn ID where the action
        occurs.
    """
    params_by_service = defaultdict(lambda: defaultdict(list))
    for turn_idx, turn in enumerate(
        dialogue_iterator(dialogue, user=user_turns, system=system_turns),
    ):
        this_turn_params = defaultdict(dict)
        turn_idx = get_turn_idx(turn_idx, user_flag=user_turns, sys_flag=system_turns)
        turn_actions_by_service = get_turn_actions(
            turn, act_patterns=act_patterns, use_lowercase=config.use_lowercase
        )  # type: dict[str, list[str]]
        merge_values = config.multiple_values_for_slot_in_turn.merge
        process_multiple_values(turn_actions_by_service, merge_values)
        map_action_params_to_canonical_vals(
            turn_actions_by_service,
            goal,
            config.value_matching,
            tracker=tracker,
            canonical_map=canonical_map,
            metadata_fields=metadata_fields,
        )

        for service in turn_actions_by_service:
            for action in turn_actions_by_service[service]:
                # actions without parameters
                try:
                    s_slot_value = action.split(ACT_SLOT_SEP)[1]
                except IndexError:
                    continue
                this_turn_params[service][s_slot_value] = [turn_idx]
        if this_turn_params:
            append_to_values(params_by_service, this_turn_params)
    return params_by_service


def is_request_action(action: str):
    for pattern in REQUEST_ACT_PATTERNS:
        if re.search(pattern, action):
            return True


def is_general_action(action: str) -> bool:
    """Relevant for MultiWOZ, where a general domain is defined for e.g., dialogue formalities."""
    return "general" in action


def has_slot_value(action: str) -> bool:
    """Heuristic function to check if a MultiWOZ action admits parameters."""
    return all(
        (
            not is_general_action(action),
            not is_request_action(action),
            SLOT_VALUE_SEP in action,
            ACT_SLOT_SEP in action,
        )
    )


def map_action_params_to_canonical_vals(
    turn_actions_by_service: dict[str, list[str]],
    goal: dict,
    config,
    tracker: Optional = None,
    canonical_map: Optional[dict] = None,
    metadata_fields: Optional[list[str]] = None,
):
    """Maps the parameters of the actions in the input action dictionary to canonical form. Input dictionary is
    modified in place. See `preprocess_constraint_values` for implementation details.
    """

    # TODO: TEST THIS FUNCTION

    if not config.use_fuzzy_match:
        return

    slot_value_pairs_by_service = defaultdict(dict)
    act_index_pair = defaultdict(list)

    # find actions that take slot value pairs
    for service, service_actions in turn_actions_by_service.items():
        for action_idx, action in enumerate(service_actions):
            # if any(re.search(pattern, action) for pattern in INFORM_ACT_PATTERNS) and ACT_SLOT_SEP not in action:
            #     logging.warning(f"Found inform action without parameters...")
            if has_slot_value(action):
                act, act_params = action.split(ACT_SLOT_SEP)
                # remove act so that the mapping function works correctly
                act_index_pair[service].append((action_idx, act))
                assert isinstance(act_params, str)
                slot_value_pairs_by_service[service][act_params] = [action_idx]
    # map values to canonical value by modifying  slot_value_pairs_by_service in place
    preprocess_constraint_values(
        slot_value_pairs_by_service,
        goal,
        config,
        tracker=tracker,
        canonical_map=canonical_map,
        metadata_fields=metadata_fields,
    )
    # replace action parameters with their canonical form
    for service in turn_actions_by_service:
        actions_to_replace = act_index_pair[service]
        for action_idx, act in actions_to_replace:
            for act_param, idx in slot_value_pairs_by_service[service].items():
                if idx == [action_idx]:
                    canonical_action = f"{act}{ACT_SLOT_SEP}{act_param}"
                    turn_actions_by_service[service][action_idx] = canonical_action


def get_system_nlu_params(
    sys_turns: list[dict],
    dial_id: str,
    goal: dict,
    tracker,
    config: DictConfig,
    canonical_map: Optional[dict] = None,
) -> dict[str, dict[str, list[int]]]:
    """Wrapper function to help retrieve system NLU parameters using the `get_parameters_by_service`.
     function.

    Parameters
    ---------
    sys_turns
        System turns for which NLU parameters are to be extracted.
    dial_id
        ID of the dialogue from which NLU is extracted. Used to store value preprocessing results under
        the key ["igcd_f1"]["preprocessed_sys_nlu"][dial_id] of the output.
    goal, tracker, config, canonical_map
        See `get_parameters_by_service` documentation.

    Returns
    -------
    sys_nlu_by_service
        A mapping containing dictionaries mapping action parameters to the index of the turns where they
        occur. Action parameters are stings formatted as "slot{SLOT_VALUE_SEP}value".
    """
    sys_nlu_turns = [
        (
            {
                "frames": turn.get("nlu", {}).get("frames", []),
                "speaker": SYSTEM_AGENT,
            },
            {"speaker": USER_AGENT},
        )
        for turn in reversed(sys_turns)
    ]
    sys_nlu_turns = [{"speaker": USER_AGENT}] + list(chain(*sys_nlu_turns))
    nlu_dial_span = {
        "turns": sys_nlu_turns,
    }
    sys_nlu_by_service = get_parameters_by_service(
        nlu_dial_span,
        goal,
        config,
        tracker,
        user_turns=False,
        act_patterns=INFORM_ACT_PATTERNS,
        canonical_map=canonical_map,
        metadata_fields=["igcd_f1", "preprocessed_sys_nlu", dial_id],
    )
    return sys_nlu_by_service


def _multiple_domains_in_turn(turn: dict) -> bool:
    """Detect if the current turn has multiple domains."""
    actions_by_service = get_turn_actions(turn)
    with suppress(KeyError):
        actions_by_service.pop("general")
    return len(actions_by_service.keys()) > 1


def _get_span_domains(dial_span: list[dict]) -> set[str]:
    """Return a list of domains mentioned in a dialogue span."""
    domains_discussed = set()
    for turn in dial_span:
        actions_by_service = get_turn_actions(turn)
        domains_discussed.update(actions_by_service.keys())
    return domains_discussed

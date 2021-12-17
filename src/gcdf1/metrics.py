from __future__ import annotations

import logging
from collections import defaultdict
from contextlib import suppress
from itertools import chain
from typing import Optional

from more_itertools import pairwise
from omegaconf import DictConfig

from gcdf1.behaviours.multiwoz import (
    is_behavioural_repetition,
    is_ref_slot,
    is_system_elicited_repetition,
    is_system_triggered_repeated_request,
    notifies_void_result,
    system_elicited_new_constraint,
    system_offered_choice,
    taxi_active_intent,
)
from gcdf1.multiwoz_metadata import (
    DOMAIN_CONSTRAINT_GOAL_KEYS,
    DOMAIN_REQUEST_GOAL_KEY,
    ENTITY_SLOTS_BY_SERVICE,
    INFORM_ACT_PATTERNS,
    SLOT_VALUE_ACT_PATTERNS,
)
from gcdf1.utils.data import get_turn_by_idx
from gcdf1.utils.dialogue import SLOT_VALUE_SEP, get_requestables_by_service, get_slots
from gcdf1.utils.evaluator import asdict, compute_f1, store_f1
from gcdf1.utils.metrics import (
    _get_span_domains,
    _multiple_domains_in_turn,
    get_parameters_by_service,
    get_system_nlu_params,
)
from gcdf1.utils.utils import append_to_values, count_odd, dispatch_on_value, safeget

REQUESTED_SLOTS_F1 = "rgcd_f1"
REQUESTED_SLOTS_PRECISION = "rgcd_precision"
REQUESTED_SLOTS_RECALL = "rcgd_recall"
REQUESTED_METRICS = [
    REQUESTED_SLOTS_F1,
    REQUESTED_SLOTS_PRECISION,
    REQUESTED_SLOTS_RECALL,
]
INFORM_F1 = "igcd_f1"
INFORM_PRECISION = "igcd_precision"
INFORM_RECALL = "igcd_recall"
INFORM_METRICS = [INFORM_F1, INFORM_PRECISION, INFORM_RECALL]
USER_METRICS = [
    REQUESTED_SLOTS_F1,
    REQUESTED_SLOTS_PRECISION,
    REQUESTED_SLOTS_RECALL,
    INFORM_F1,
    INFORM_PRECISION,
    INFORM_RECALL,
]
VALID_AGENT_VALUES = ["USER", "SYSTEM", "BOTH"]
"""Valid values for the `agent` setting in the configuration file."""
_SPECIAL_METADATA_KEYS = {
    "nobook_validation_failure",
    "wrong_taxi_annotations",
    "nooffer_validation_failure",
    "failed_dialogue",
    "repeated_despite_receiving_answer",
}
NAN_VAL = "NA"


def is_transactional_goal(domain_goal: dict):
    """Checks if a domain goal is transactional. For MultiWOZ this is equivalent to the presence of a ``'book'``
    field in the domain goal.

    Parameters
    ----------
    domain_goal
        Domain goal containing constraints and information requests. See `compute_igcd_f1` for full goal structure.

    """
    return "book" in domain_goal and domain_goal["book"]


def get_domain_requests(
    goal: dict, domain: str, use_lowercase: bool = True
) -> set[str]:
    """Retrieve the requested slots for `domain` from the goal.

    Parameters
    ----------
    goal
        The user goal for the dialogue.
    domain
        Domain for which requests are to be returned.
    use_lowercase
        If ``True``, lowercase slot names.
    """
    domain_goal = goal.get(domain, {})
    # difference between convlab and multiwoz goal format
    try:
        domain_true_pos = set(domain_goal.get(DOMAIN_REQUEST_GOAL_KEY, {}).keys())
    except AttributeError:
        domain_true_pos = set(domain_goal.get(DOMAIN_REQUEST_GOAL_KEY, []))
    if use_lowercase:
        return {slot.lower() for slot in domain_true_pos}
    return domain_true_pos


def _detect_spurious_annotations(
    request: str, sys_turns: list[dict], usr_turns: list[dict], domain: str
) -> bool:
    """Heuristic rules for detecting spurious request annotations."""

    excluded_domains = ["booking", "general"]
    if is_ref_slot(request):
        return False
    if domain == "taxi" and domain in request:
        return False
    span_domains = _get_span_domains(sys_turns)
    for val in excluded_domains:
        try:
            span_domains.remove(val)
        except KeyError:
            pass
    if not span_domains:
        return False
    if domain not in span_domains and len(sys_turns) > 1:
        return True
    else:
        multiple_domains = [_multiple_domains_in_turn(turn) for turn in usr_turns]
        if any(multiple_domains) and len(sys_turns) > 3 and span_domains != {domain}:
            return True
    return False


def get_previous_sys_turns(
    dialogue: dict, start_turn_idx: int, context_size: Optional[int] = -1
) -> list[dict]:
    """Retrieve previous system turns.

    Parameters
    ----------
    dialogue
    start_turn_idx
        The user turn before which system turns are considered. Should be divisible by 2.
    context_size
        How many system turns to return.

    Returns
    -------
    sys_turns
        System turn dictionaries.

    """
    assert start_turn_idx % 2 == 0
    sys_turn_lb = -1
    if context_size > 0:
        sys_turn_lb = max(-1, start_turn_idx - 1 - 2 * context_size)
    assert sys_turn_lb % 2 == 1
    sys_turn_idx = range(start_turn_idx - 1, sys_turn_lb, -2)
    assert list(sys_turn_idx)
    sys_turns = [get_turn_by_idx(dialogue, idx) for idx in sys_turn_idx]
    assert sys_turns
    return sys_turns


def compute_rgcdf1(dialogue: dict, goal: dict, tracker, config: DictConfig) -> dict:
    """Computes the goal- and context- drive requested slots F1 score (R-GCDF1).

    Parameters
    ----------
    dialogue
        Dialogue to evaluate, in SGD format.
    goal
        Goal of the dialogue, in the following format::
            {
                'domain': {
                        'fail_info': {'slot_name': value, ....}
                        'info': {'slot_name': value, ....}
                        DOMAIN_REQUEST_GOAL_KEY: {'slot_name': ? ...}
                        'book': {'slot_name': value, ....}
                }

            }
        Only the ``DOMAIN_REQUEST_GOAL_KEY`` field in each domain is used for this metric. The field should not be
        included for domains where there are no requested slots.
    config
        A configuration defining the user/system behaviours taken into account in evaluation and hyperparameters. See
        configs/multiwoz_user_evaluator.yaml, field 'rgcd_f1' for configuration options.
    tracker
        A nested defaultdict which stores information during R-GCDF1 calculation, which can be used to understand
        agent behaviour.
    """

    def store_metadata(info, metadata_fields: list[str]):

        nonlocal metadata
        store_location = safeget(metadata, *metadata_fields)
        if isinstance(store_location, list):
            store_location.append(info)
        else:
            raise NotImplementedError

    metadata = getattr(tracker, "metadata")["rgcd_f1"]
    dial_id = dialogue["dialogue_id"]
    count_repetitions = config.repetitions.count_fp
    auto_matched_requests = list(config.auto_add_to_ref)
    # extract user requested slots from dialogue
    # the hypothesised request slots are formatted as "domain-slot"
    hyp_user_requests = get_requestables_by_service(
        dialogue, agent="USER", use_lowercase=config.use_lowercase
    )  # type: dict[str, dict[str, list[int]]]
    # do not penalise the user if the system informs slots that should be requested
    # before the user asks the question (this happens in the corpus in taxi domain, a lot)
    # TODO: IMPROVE READABILITY!
    if config.missing_requests.system_provision:
        for domain in config.missing_requests.provision_domains:
            domain_key = [dial_id, domain]
            domain_true_pos = get_domain_requests(
                goal, domain, use_lowercase=config.use_lowercase
            )
            if not domain_true_pos:
                continue
            # TODO: TEST THIS FUNCTION
            domain_sys_provisioned = get_slots(
                dialogue,
                agent="SYSTEM",
                act_patterns=SLOT_VALUE_ACT_PATTERNS,
                service_patterns=[domain, "booking"],
            )
            provisioned_in_goal, turn_provisioned = [], []
            for dom in domain_sys_provisioned:
                for slot in domain_sys_provisioned[dom]:
                    if slot in domain_true_pos:
                        provisioned_in_goal.append(f"{domain}-{slot}")
                        turn_provisioned.append([domain_sys_provisioned[dom][slot][0]])
            # user has not had time to request any slots
            if domain not in hyp_user_requests:
                if provisioned_in_goal:
                    field = ["provision_entire_domain"] + domain_key
                    store_metadata(provisioned_in_goal, field)
                    append_to_values(
                        hyp_user_requests,
                        {domain: dict(zip(provisioned_in_goal, turn_provisioned))},
                    )
            # some slots were requested by the user and we add only the additional
            # ones that were not but were mentioned by the sys only
            else:
                outstanding_provisioned, outstanding_provisioned_turn_idx = [], []
                for idx, request in enumerate(provisioned_in_goal):
                    if request not in hyp_user_requests[domain]:
                        outstanding_provisioned.append(request)
                        outstanding_provisioned_turn_idx.append(turn_provisioned[idx])
                if outstanding_provisioned:
                    field = ["provision_slots"] + domain_key
                    store_metadata(outstanding_provisioned, field)
                    append_to_values(
                        hyp_user_requests,
                        {
                            domain: dict(
                                zip(
                                    outstanding_provisioned,
                                    outstanding_provisioned_turn_idx,
                                )
                            )
                        },
                    )

    true_pos, pos = [], []
    # compute metrics for each domain in goal
    goal_domains = set(goal.keys())
    for domain in goal_domains:
        domain_key = [dial_id, domain]
        domain_true_pos = get_domain_requests(
            goal, domain, use_lowercase=config.use_lowercase
        )
        # format slots consistently with the requested slots repr, which is domain-slot
        domain_true_pos = [f"{domain}-{slot}" for slot in domain_true_pos]
        domain_pos = []
        if domain in hyp_user_requests:
            for request in hyp_user_requests[domain]:
                n_repetitions = len(hyp_user_requests[domain][request])
                if n_repetitions > 1:
                    field = ["multiple_req"] + domain_key
                    store_metadata({request: n_repetitions}, field)
                if n_repetitions > config.repetitions_warning:
                    store_metadata(request, ["repetitions_warning"] + domain_key)
                    logging.warning(
                        f"User requested slot {request} {n_repetitions} times"
                    )
                if not count_repetitions:
                    n_repetitions = 1
                domain_pos += [request] * n_repetitions
                if request not in domain_true_pos and request in auto_matched_requests:
                    # compensate for slots missing in domain goal annotations
                    if is_transactional_goal(goal[domain]):
                        domain_true_pos.append(request)
                        store_metadata(dial_id, ["auto_ref", domain])
                if count_repetitions and n_repetitions > 1:
                    store_key = ["multiple_slot_requests"]
                    # TODO: THIS PART OF THE CODE AND/OR-AT-LEAST FUNCTIONS IT RELIES ON NEED TO BE TESTED
                    # TODO: LONG TERM: VALIDATE SYS NLG AS WELL
                    max_repeats = config.repetitions.repeats_to_penalty
                    turns_req_made = hyp_user_requests[domain][request]
                    # if the sys NLU did not work correctly, increase true positives
                    # for each slot by up to user_max_repeats
                    sys_behaviour_config = config.repetitions.system_behaviour
                    for repetition, repetition_span in enumerate(
                        pairwise(turns_req_made), start=1
                    ):
                        turn_idx, rep_turn_idx = repetition_span
                        # requested slots represented in domain-slot format
                        assert "-" in request
                        usr_turns = [
                            get_turn_by_idx(dialogue, idx) for idx in repetition_span
                        ]
                        sys_turns = get_previous_sys_turns(
                            dialogue,
                            rep_turn_idx,
                            context_size=count_odd(turn_idx, rep_turn_idx),
                        )
                        if _detect_spurious_annotations(
                            request, sys_turns, usr_turns, domain
                        ):
                            field = store_key + ["spurious_annotation"] + domain_key
                            store_metadata({request: list(repetition_span)}, field)
                            continue
                        sys_behaviours = is_system_triggered_repeated_request(
                            sys_turns, request, domain, sys_behaviour_config
                        )  # type: list[str]

                        # if there is an answer in the sys turn span, penalise user
                        if _SPECIAL_METADATA_KEYS.intersection(sys_behaviours):
                            for behaviour in sys_behaviours:
                                field = store_key + [behaviour] + domain_key
                                store_metadata({request: n_repetitions}, field)
                        else:
                            if sys_behaviours:
                                if repetition <= max_repeats:
                                    domain_true_pos.append(request)
                                for behaviour in sys_behaviours:
                                    field = store_key + [behaviour] + domain_key
                                    store_metadata({request: n_repetitions}, field)

        pos += domain_pos
        true_pos += domain_true_pos

        if not domain_true_pos:
            if domain_pos:
                logging.warning(
                    f"{dialogue['dialogue_id']}: No requested slots in goal but the user requested slots."
                )
                store_metadata(domain_pos, ["requests_not_in_goal", dial_id, domain])
        # this happens if there are no requests for a given domain
        if not domain_pos and not domain_true_pos:
            continue
        dom_goal_f1_scores = compute_f1(domain_true_pos, domain_pos)
        store_f1(
            REQUESTED_METRICS,
            tracker,
            dom_goal_f1_scores,
            metric_key=domain,
            dial_id=dial_id,
        )

    # check the user does not request slots in domains that are not in goal
    user_only_domains = set(hyp_user_requests.keys()) - goal_domains
    for domain in user_only_domains:
        user_only_dom_pos = []
        for request in hyp_user_requests[domain]:
            n_repetitions = (
                len(hyp_user_requests[domain][request])
                if config.repetitions.count_fp
                else 1
            )
            user_only_dom_pos += [request] * n_repetitions
        f1_scores = compute_f1([], user_only_dom_pos)
        store_f1(
            REQUESTED_METRICS,
            tracker,
            f1_scores,
            metric_key=domain,
            dial_id=dial_id,
        )
        pos += user_only_dom_pos
        logging.warning(
            f"{dialogue['dialogue_id']}: User requested slots from domain {domain}, not present in goal"
        )
        store_metadata(
            user_only_dom_pos, ["requested_domain_outside_goals", dial_id, domain]
        )
    # compute metrics disregarding domain if any slots were requested in the dialogue
    if true_pos:
        combined_f1 = compute_f1(true_pos, pos)
        store_f1(
            REQUESTED_METRICS,
            tracker,
            combined_f1,
            metric_key="combined",
            dial_id=dial_id,
        )
    if not true_pos:
        if pos:
            logging.warning(
                f"{dialogue['dialogue_id']}: User requested slots {pos} but no slots provided in goal."
            )
            combined_f1 = compute_f1(true_pos, pos)
            store_f1(
                REQUESTED_METRICS,
                tracker,
                combined_f1,
                metric_key="combined",
                dial_id=dial_id,
            )

    metadata["pos"] = pos
    metadata["true_pos"] = true_pos
    return asdict(tracker)


@dispatch_on_value
def _is_auto_matched(domain: str, constraint: str, config) -> list[str]:
    """The default functionality is to check if the constraint contains a value that is automatically matched
    (e.g., dontcare) if the slot has not been included in the user goal, for all domains except taxi.
    """
    auto_matched_values = config.vals  # type: list[str]
    slot, value = constraint.split(SLOT_VALUE_SEP)
    # nb: here we don't check if the constraint was requested: we assume the user is
    #  free to inform any constraints not in goal with wildcard values if they are appropriate
    # See MUL2630 as an example.
    if value in auto_matched_values:
        return ["auto_matched_value"]
    auto_matched_slot_values = (
        config.slot_value_pairs
    )  # type: dict[str, dict[str, list[str]]]
    auto_matched_slot_values = auto_matched_slot_values.get(domain, {}).get(slot, [])
    if value in auto_matched_slot_values:
        return ["auto_matched_slot_value_pair"]
    return []


@_is_auto_matched.register("taxi")
def _(domain: str, constraint: str, config) -> list[str]:
    """
    Marks taxi domain departure/destination constraint as valid as they are not specified in goal.
    """

    auto_matched_values = config.vals  # type: list[str]
    auto_matched_slots = config.slots.get(domain, {})
    slot, value = constraint.split(SLOT_VALUE_SEP)
    if value in auto_matched_values:
        return ["auto_matched_value"]
    if slot in auto_matched_slots:
        return ["auto_matched_slot"]
    if not config.slot_value_pairs:
        return []
    auto_matched_slot_values = (
        config.slot_value_pairs
    )  # type: dict[str, dict[str, str]]
    auto_matched_slot_values = auto_matched_slot_values.get(domain, {}).get(slot, {})
    if value in auto_matched_slot_values:
        return ["auto_matched_slot_value_pair"]
    return []


def get_goal_constraints(
    goal: dict, use_lowercase: bool = True
) -> dict[str, list[str]]:
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
                'domain': list[str], where each element is a constraint
            }

        Each constraint is formatted as "{slot}{SLOT_VALUE_SEP}{value}"

    Notes:
    -----
        The extraction proceeds by extracting all the slots and values from the
        keys specified in `DOMAIN_CONSTRAINT_GOAL_KEYS` and the keys of the output
        contain duplicates.
    """

    goal_constraints_by_service = defaultdict(list)

    for domain in goal:
        # extract information from domain sub-goals (except reqt)
        # this will contain duplicates depending on fail_* subgoals content
        for subgoal_key in DOMAIN_CONSTRAINT_GOAL_KEYS:
            subgoal_slot_value_map = goal[domain].get(subgoal_key, {})
            for name, value in subgoal_slot_value_map.items():
                if use_lowercase:
                    name, value = name.lower(), value.lower()
                goal_constraints_by_service[domain].append(
                    f"{name}{SLOT_VALUE_SEP}{value}"
                )

    return dict(goal_constraints_by_service)


def _detect_empty_db_response(
    domain: str,
    hyp_user_constraints: dict[str, dict[str, list[int]]],
    domain_true_pos: list[str],
    missing_constraints: set[str],
    dialogue: dict,
) -> list[tuple[str, str]]:
    """Detect if missing constraints are not provided due to an empty database response. This is MultiWOZ specific."""

    behaviours = []
    if missing_constraints:
        logging.warning(
            f"{dialogue['dialogue_id']}: User did not express constraints {missing_constraints}..."
        )
        provided_constraints = hyp_user_constraints[domain]
        # if the user informed the same slot, it means it had
        # a different value so even if the dial failed, it
        # is because of user's fault
        for constr in missing_constraints:
            missing_slot, _ = constr.split(SLOT_VALUE_SEP)
            for prov_constr in provided_constraints:
                # this will also detect if day slot was informed in other slots (e.g., leaveat=thursday)
                if missing_slot in prov_constr:
                    # this means that the user informed an alternative that was in the goal
                    if prov_constr in domain_true_pos:
                        continue
                    behaviours.append(
                        (
                            "informed_wrong_value",
                            {"provided": prov_constr, "expected": constr},
                        )
                    )
                    break
            else:
                behaviours.append(("missed_constraints", constr))

        # no constraint
        if not provided_constraints:
            return behaviours

        # if the user did not inform a slot, check to see if the DB
        # returned no results after they informed their last constraint
        constraint_turn_idx = list(chain(*provided_constraints.values()))
        sys_turn = get_turn_by_idx(
            dialogue,
            max(constraint_turn_idx) + 1,
        )
        void_result = notifies_void_result(domain, sys_turn, use_lowercase=True)
        choice_offered = system_offered_choice(domain, sys_turn, use_lowercase=True)
        if void_result and not choice_offered:
            logging.warning(f"{dialogue['dialogue_id']}: appears to have failed ....")
            behaviours = [("failed_dialogue", constr) for constr in missing_constraints]
    return behaviours


def offset_turn_idx(params_by_service: dict[str, dict[str, list[int]]], start: int):
    """Offsets turn index of the input action parameters by a fixed value so that the
    parameters extracted from an arbitrary dialogue span carry correct turn information.
    """
    for service in params_by_service:
        for slot_value in params_by_service[service]:
            params_by_service[service][slot_value] = [
                start + val for val in params_by_service[service][slot_value]
            ]


def check_nlu_parsing(
    sys_nlu_by_service: dict[str, dict[str, list[int]]],
    repetition_span: tuple[int, int],
):
    """Sanity check that turn indices in parsed dialogue spans are correct."""
    turn_idx, rep_turn_idx = repetition_span
    for domain in sys_nlu_by_service:
        for constraint in sys_nlu_by_service[domain]:
            t_idx = sys_nlu_by_service[domain][constraint]
            assert all(
                idx % 2 == 1 and idx in range(turn_idx, rep_turn_idx) for idx in t_idx
            )


def compute_igcdf1(
    dialogue: dict, goal: dict, tracker, config, canonical_map: Optional[dict] = None
) -> dict:
    """Computes the goal- and context- driven Inform F1 score (I-GCDF1).

    Parameters
    ----------
    dialogue
        Dialogue to evaluate, in SGD format.
    goal
        Goal of the dialogue, in the following format::
            {
                'domain_name': {
                        'sub_goal_key_1': {'slot_name': value, ....},
                        'sub_goal_key_2': {'slot_name': value, ....},
                        ...
                },
                ...
            }
        The sub-goal keys 'sub_goal_key_*' are defined in DOMAIN_CONSTRAINT_GOAL_KEYS. See documentation for this
        constant in `multiwoz_metadata.py` for MultiWOZ 2.1. corpus.
    canonical_map
        Corpus-dependent paraphrase table. See resources/multiwoz/multiwoz21_canonical_map.json for a MultiWOZ 2.1
        example.
    config
        A configuration defining the user/system behaviours taken into account in evaluation and hyperparameters. See
        configs/multiwoz_user_evaluator.yaml, field 'igcd_f1' for configuration options.
    tracker
        A nested defaultdict which stores information during I-GCDF1 calculation, which can be used to understand
        agent behaviour.
    """

    def store_metadata(info, metadata_fields: list[str]):

        nonlocal metadata
        store_location = safeget(metadata, *metadata_fields)
        if isinstance(store_location, list):
            store_location.append(info)
        else:
            raise NotImplementedError

    dial_id = dialogue["dialogue_id"]
    metadata = getattr(tracker, "metadata")["igcd_f1"]
    outside_goal_config = config.constraints_not_in_goal
    repetitions_behaviour = config.repetitions
    count_repetitions = repetitions_behaviour.count_fp
    hyp_user_constraints = get_parameters_by_service(
        dialogue,
        goal,
        config,
        tracker=tracker,
        system_turns=False,
        act_patterns=INFORM_ACT_PATTERNS,
        canonical_map=canonical_map,
        metadata_fields=["igcd_f1", "preprocessed_user_constraints", dial_id],
    )  # type: dict[str, dict[str, list[int]]]
    goal_constraints = get_goal_constraints(goal)  # type: dict[str, list[str]]

    true_pos, pos = [], []
    goal_domains = set(goal.keys())
    # compute the F1 score for each domain
    for domain in goal_domains:
        domain_true_pos, domain_pos = goal_constraints.get(domain, []), []

        domain_goal_slots = {
            constr.split(SLOT_VALUE_SEP)[0] for constr in domain_true_pos
        }
        # eliminate duplicate slots in info/fail info as domain_true_pos
        # is updated to reflect repreated constraints if database/transactions fail
        domain_true_pos = list(set(domain_true_pos))
        # extract information from user policy output. If domain is missed, then
        # domain_pos will not be be updated so the user will be penalised
        if domain in hyp_user_constraints:
            domain_key = [dial_id, domain]
            for constraint in hyp_user_constraints[domain]:
                turns_constraint_informed = hyp_user_constraints[domain][constraint]
                n_repetitions = len(turns_constraint_informed)
                if n_repetitions > 1:
                    field = ["repeated_inform"] + domain_key
                    store_metadata({constraint: n_repetitions}, field)
                if n_repetitions > config.repetitions_warning:
                    store_metadata(constraint, ["repetitions_warning"] + domain_key)
                    logging.warning(
                        f"User repeated constraint {constraint} {n_repetitions} times"
                    )
                if not count_repetitions:
                    n_repetitions = 1
                domain_pos += [constraint] * n_repetitions
                slot, val = constraint.split(SLOT_VALUE_SEP)
                assert isinstance(val, str)
                if slot not in domain_goal_slots:
                    # automatically match some constraints that are not expressed in the goals
                    store_key = ["constraints_not_in_goal"]
                    auto_matched = _is_auto_matched(
                        domain, constraint, outside_goal_config.auto_matching
                    )
                    if auto_matched:
                        # matching a given number of repetitions
                        domain_true_pos += [constraint] * min(
                            n_repetitions,
                            outside_goal_config.auto_matching.max_repetions_matched,
                        )
                        assert len(auto_matched) == 1
                        field = store_key + auto_matched + domain_key
                        store_metadata({constraint: n_repetitions}, field)
                        # deal with the next constraint, this one is auto-matched
                        continue
                    else:
                        max_repetitions = outside_goal_config.max_repetitions
                        for repetition, turn_idx in enumerate(
                            turns_constraint_informed, start=1
                        ):
                            constraint_info = {constraint: turn_idx}
                            assert turn_idx % 2 == 0
                            if turn_idx == 0:
                                field = store_key + ["first_turn"] + domain_key
                                store_metadata(constraint_info, field)
                            else:
                                sys_turns = get_previous_sys_turns(
                                    dialogue,
                                    turn_idx,
                                )
                                sys_elicited_behaviours = (
                                    system_elicited_new_constraint(
                                        domain,
                                        constraint,
                                        sys_turns,
                                        outside_goal_config.sys_checks,
                                    )
                                )
                                if sys_elicited_behaviours:
                                    # log wrong annotations for taxi domain
                                    if slot in ENTITY_SLOTS_BY_SERVICE[domain]:
                                        this_turn = get_turn_by_idx(dialogue, turn_idx)
                                        if (
                                            taxi_active_intent(this_turn)
                                            or "taxi" in this_turn["utterance"]
                                        ):
                                            field = ["wrong_taxi_annotations", dial_id]
                                            store_metadata(constraint_info, field)
                                            continue
                                    if repetition <= max_repetitions:
                                        domain_true_pos.append(constraint)
                                    for behaviour in sys_elicited_behaviours:
                                        field = store_key + [behaviour] + domain_key
                                        store_metadata(constraint_info, field)
                                    if not count_repetitions:
                                        break
                                else:
                                    field = store_key + ["unmatched"] + domain_key
                                    store_metadata(constraint_info, field)
                        # rest of the code handles constraints in goal
                        continue
                # increase the counts in domain_true_pos if the constraint was
                # repeated for a legitimate reason
                if count_repetitions and n_repetitions > 1:
                    store_key = ["multiple_constraint_informs"]
                    max_repetitions = repetitions_behaviour.max_repetitions
                    turns_repeated = hyp_user_constraints[domain][constraint]
                    for repetition, (turn_idx, rep_turn_idx) in enumerate(
                        pairwise(turns_repeated), start=1
                    ):
                        sys_turns = get_previous_sys_turns(
                            dialogue,
                            rep_turn_idx,
                            context_size=count_odd(turn_idx, rep_turn_idx),
                        )

                        sys_nlu_by_service = {}
                        if repetitions_behaviour.sys_behaviours.nlu:
                            sys_nlu_by_service = get_system_nlu_params(
                                sys_turns, dial_id, goal, tracker, config, canonical_map
                            )
                            offset_turn_idx(sys_nlu_by_service, start=turn_idx)
                            check_nlu_parsing(
                                sys_nlu_by_service, (turn_idx, rep_turn_idx)
                            )

                        sys_elicited_behaviours = is_system_elicited_repetition(
                            domain,
                            constraint,
                            sys_turns,
                            repetitions_behaviour.sys_behaviours,
                            sys_nlu=sys_nlu_by_service,
                        )
                        special_keys = [
                            key
                            for key in _SPECIAL_METADATA_KEYS
                            if key in sys_elicited_behaviours
                        ]
                        constraint_info = {constraint: [turn_idx, rep_turn_idx]}
                        # some behaviours are actually checks that rules work,
                        # are removed before considering whether to compensate
                        for key in special_keys:
                            sys_elicited_behaviours.remove(key)
                            field = store_key + [key] + domain_key
                            store_metadata(constraint_info, field)
                        if sys_elicited_behaviours:
                            if repetition <= max_repetitions:
                                domain_true_pos.append(constraint)
                            for behaviour in sys_elicited_behaviours:
                                field = store_key + [behaviour] + domain_key
                                store_metadata(constraint_info, field)
                            # TODO: LOG WRONG TAXI ANNOTATIONS HERE ? (MUL0011, MUL1351)
                            # no point in checking user behaviours, we know why the
                            # repetition occurred
                            continue
                        usr_turn = get_turn_by_idx(dialogue, rep_turn_idx)
                        user_behaviours = is_behavioural_repetition(
                            domain,
                            constraint,
                            usr_turn,
                            sys_turns,
                            repetitions_behaviour.user_behaviours,
                        )
                        if user_behaviours:
                            if "wrong_taxi_annotations" in user_behaviours:
                                field = (
                                    store_key + ["wrong_taxi_annotations"] + domain_key
                                )
                                store_metadata(constraint_info, field)
                                continue
                            if repetition <= max_repetitions:
                                domain_true_pos.append(constraint)
                            for behaviour in user_behaviours:
                                field = store_key + [behaviour] + domain_key
                                store_metadata(constraint_info, field)
                        else:
                            field = store_key + ["unmatched"] + domain_key
                            store_metadata(constraint_info, field)
                elif n_repetitions == 1:
                    # these are likely noisy dialogues. Checked from bottom to SNG0616
                    if constraint not in domain_true_pos:
                        # this field is a superset of 'informed_wrong_value'
                        # which contains dialogues where the annotation was
                        # wrong but the system mentioned the correct value later on
                        field = ["single_repetitions_nog"] + domain_key
                        constraint_info = {
                            constraint: hyp_user_constraints[domain][constraint][0]
                        }
                        store_metadata(constraint_info, field)

        # some constraints might be missing because the system
        # mentioned them
        if not domain_pos and domain_true_pos:
            field = ["missed_domain", dial_id]
            store_metadata(domain, field)
        missing_constraints = set(domain_true_pos) - set(domain_pos)
        if missing_constraints:
            sys_slot_value_pairs = get_parameters_by_service(
                dialogue,
                goal,
                config,
                tracker=tracker,
                user_turns=False,
                act_patterns=SLOT_VALUE_ACT_PATTERNS,
                canonical_map=canonical_map,
                metadata_fields=["igcd_f1", "preprocessed_sys_constraints", dial_id],
            )
        system_offered_constraints = set()
        for constraint in missing_constraints:
            if (
                constraint in sys_slot_value_pairs[domain]
                or constraint in sys_slot_value_pairs["booking"]
            ):
                if constraint in sys_slot_value_pairs[domain]:
                    turns = sys_slot_value_pairs[domain][constraint]
                else:
                    turns = sys_slot_value_pairs["booking"][constraint]
                system_offered_constraints.add(constraint)
                field = ["sys_preempted_constraint"] + [dial_id, domain]
                store_metadata({constraint: turns}, field)
                if config.compensate_preemption:
                    domain_pos.append(constraint)
        missing_constraints.difference_update(system_offered_constraints)

        dial_status = _detect_empty_db_response(
            domain, hyp_user_constraints, domain_true_pos, missing_constraints, dialogue
        )
        fail_config = config.empty_db_result
        if dial_status:
            for item in dial_status:
                store_key, constraint = item
                field = [store_key, dial_id, domain]
                store_metadata(constraint, field)
                failed_dial = store_key in _SPECIAL_METADATA_KEYS
                if failed_dial and fail_config.compensate_request:
                    with suppress(KeyError):
                        goal[domain].pop(DOMAIN_REQUEST_GOAL_KEY, {})
                if failed_dial and fail_config.compensate_missed_constraints:
                    domain_true_pos.remove(constraint)

        true_pos += domain_true_pos
        pos += domain_pos

        if not domain_true_pos:
            if domain_pos:
                logging.warning(
                    f"{dial_id}: No inform slots in goal but the user informed slots: {domain_pos}"
                )
        # this happens for certain domains (police, hospital in MultiWOZ)
        if not domain_pos and not domain_true_pos:
            continue
        dom_goal_f1_scores = compute_f1(domain_true_pos, domain_pos)
        store_f1(
            INFORM_METRICS,
            tracker,
            dom_goal_f1_scores,
            metric_key=domain,
            dial_id=dialogue["dialogue_id"],
        )

    # check the user does not inform constraints from domains not in goal
    user_only_domains = set(hyp_user_constraints.keys()) - goal_domains
    for domain in user_only_domains:
        user_only_dom_pos = []
        for slot_value in hyp_user_constraints[domain]:
            n_repetitions = (
                len(hyp_user_constraints[domain][slot_value])
                if count_repetitions
                else 1
            )
            user_only_dom_pos += [slot_value] * n_repetitions
        metadata["informed_domains_outside_goals"][dial_id][domain] += user_only_dom_pos

        f1_scores = compute_f1([], user_only_dom_pos)
        store_f1(
            INFORM_METRICS,
            tracker,
            f1_scores,
            metric_key=domain,
            dial_id=dialogue["dialogue_id"],
        )
        pos += user_only_dom_pos

    if not true_pos:
        # sanity check, should not happen
        if pos:
            logging.warning(
                f"{dial_id}: Goal contains no constraints but the user informed constraints {pos}."
            )
            combined_f1 = compute_f1(true_pos, pos)
            store_f1(
                INFORM_METRICS,
                tracker,
                combined_f1,
                metric_key="combined",
                dial_id=dialogue["dialogue_id"],
            )
    else:
        combined_f1 = compute_f1(true_pos, pos)
        store_f1(
            INFORM_METRICS,
            tracker,
            combined_f1,
            metric_key="combined",
            dial_id=dialogue["dialogue_id"],
        )

    metadata["pos"] = pos
    metadata["true_pos"] = true_pos

    return asdict(tracker)


# TODO: CHANGE MAPPING GENERATION SO THAT TRAIN `Ref` is lowercased.

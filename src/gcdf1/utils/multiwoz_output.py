from collections import defaultdict

from gcdf1.utils.utils import nested_defaultdict

metadata_to_agg = (
    ["igcd_f1", "dial_id"],  # dialogue ID
    [
        "igcd_f1",
        "preprocessed_user_constraints",
    ],  # stores output of value pre-processing algorithm for user turns
    [
        "igcd_f1",
        "preprocessed_sys_constraints",
    ],  # stores output of value pre-processing algorithm for sys turns
    [
        "igcd_f1",
        "preprocessed_sys_nlu",
    ],  # stores output of value pre-processing algorithm for sys NLU
    [
        "igcd_f1",
        "sys_preempted_constraint",
    ],  # dials where a missing constraint is found in sys turns (see gcdf1.metrics.compute_igcdf1) # noqa
    [
        "igcd_f1",
        "single_repetitions_nog",
    ],  # dials where a constraint that is not in goal is repeated only once (see gcdf1.metrics.compute_igcdf1) # noqa
    [
        "igcd_f1",
        "missed_constraints",
    ],  # dials where a constraint was missed and not informed with a wrong value (see gcdf1.metrics._detect_empty_db_response)  # noqa
    [
        "igcd_f1",
        "informed_wrong_value",
    ],  # dials where a constraint was contained the wrong value as opposed to being completely missed (see gcdf1.metrics._detect_empty_db_response) # noqa
    [
        "igcd_f1",
        "failed_dialogue",
    ],  # dials where missing constraints occur due to the fact that the DB did not return any results (e.g., booking constraints are not expressed  # noqa
    # because a rest was not found (see gcdf1.metrics._detect_empty_db_response) # noqa
    [
        "igcd_f1",
        "missed_domain",
    ],  # dials where a domain was missed altogether by the user
    [
        "igcd_f1",
        "constraints_not_in_goal",
        "auto_matched_slot",
    ],  # dials where a constraint is not penalised because it contains a certain slot (see constraints_not_in_goal.auto_matching options and gcdf1.metrics._is_auto_matched)  # noqa
    [
        "igcd_f1",
        "constraints_not_in_goal",
        "auto_matched_value",
    ],  # as above, but for slot value
    [
        "igcd_f1",
        "constraints_not_in_goal",
        "auto_matched_slot_value_pair",
    ],  # as above, but for slot-value pair
    [
        "igcd_f1",
        "constraints_not_in_goal",
        "elicited_by_sys_question",
    ],  # dials where a constraint not in goal is due to system behaviour (see gcdf1.behaviours.multiwoz.system_elicited_new_constraint) # noqa
    [
        "igcd_f1",
        "constraints_not_in_goal",
        "elicited_by_sys_offer",
    ],  # dials where the user expresses a constraint not in goal after the system made an offer (see gcdf1.behaviours.multiwoz.system_elicited_new_constraint) # noqa
    [
        "igcd_f1",
        "constraints_not_in_goal",
        "elicited_by_booking_failure",
    ],  # dials where the user expresses a constraint not in goal after a booking failure (see gcdf1.behaviours.multiwoz.system_elicited_new_constraint) # noqa
    [
        "igcd_f1",
        "constraints_not_in_goal",
        "elicited_by_booking_acceptance",
    ],  # dials where the user expresses a constraint not in goal when they accept a sys booking offer (see gcdf1.behaviours.multiwoz.system_elicited_new_constraint)
    [
        "igcd_f1",
        "constraints_not_in_goal",
        "elicited_by_recommendation",
    ],  # dials where the user expresses a constraint not in goal after a system recommendation (see gcdf1.behaviours.multiwoz.system_elicited_new_constraint)
    [
        "igcd_f1",
        "constraints_not_in_goal",
        "first_turn",
    ],  # dials where a constraint not in goal was expressed in the first user turn
    [
        "igcd_f1",
        "constraints_not_in_goal",
        "unmatched",
    ],  # dials where a constraint not in goal was not because any of the implemented behaviours
    [
        "igcd_f1",
        "multiple_constraint_informs",
        "elicited_by_sys_question",
    ],  # dials where constraint is repeated because sys asked (see gcdf1.behaviours.multiwoz.is_system_elicited_repetition) # noqa
    [
        "igcd_f1",
        "multiple_constraint_informs",
        "elicited_by_no_offers",
    ],  # dials where constraint is repeated because there was no DB result (see gcdf1.behaviours.multiwoz.is_system_elicited_repetition) # noqa
    [
        "igcd_f1",
        "multiple_constraint_informs",
        "nobook_validation_failure",
    ],  # sanity check, this is not a behaviour  # noqa
    [
        "igcd_f1",
        "multiple_constraint_informs",
        "nooffer_validation_failure",
    ],  # sanity check, this is not a behaviour  # noqa
    [
        "igcd_f1",
        "multiple_constraint_informs",
        "confirmed_choice_params",
    ],  # dials where a constraint is repeated as a confirmation of attributes (see  gcdf1.behaviours.multiwoz.is_behavioural_repetition)  # noqa
    [
        "igcd_f1",
        "multiple_constraint_informs",
        "repeated_while_requesting_info",
    ],  # dials where a constraint is repeated as info is requested (see  gcdf1.behaviours.multiwoz.is_behavioural_repetition)         # noqa
    [
        "igcd_f1",
        "multiple_constraint_informs",
        "repeated_while_booking",
    ],  # dials where a constraint is repeated during a conversation about a transaction (see  gcdf1.behaviours.multiwoz.is_behavioural_repetition) # noqa
    [
        "igcd_f1",
        "multiple_constraint_informs",
        "repeated_when_answering_request",
    ],  # dials where a constraint is repeated when answering a sys request (see gcdf1.behaviours.multiwoz.is_behavioural_repetition) # noqa
    [
        "igcd_f1",
        "multiple_constraint_informs",
        "switched_between_domains",
    ],  # dials where a constraints is repeated to remind system of previous conversation after a different domain has been discussed (see gcdf1.behaviours.multiwoz.is_behavioural_repetition) # noqa
    [
        "igcd_f1",
        "multiple_constraint_informs",
        "wrong_taxi_annotations",
    ],  # heuristic, ignore
    [
        "igcd_f1",
        "multiple_constraint_informs",
        "unmatched",
    ],  # dials where the user repetition cannot be explained by any sys/usr behaviour  # noqa
    [
        "igcd_f1",
        "multiple_constraint_informs",
        "failed_sys_nlu",
    ],  # dials where repetitions occurred because sys did not understand the constraint (see gcdf1.behaviours.multiwoz.is_system_elicited_repetition) # noqa
    [
        "igcd_f1",
        "multiple_constraint_informs",
        "elicited_by_booking_failure",
    ],  # dials where repetitions occurred because a booking was not successful # noqa
    ["igcd_f1", "wrong_taxi_annotations"],  # heuristic, ignore
    [
        "igcd_f1",
        "repeated_inform",
    ],  # all dials where a constraint is informed more than once
    [
        "igcd_f1",
        "repetitions_warning",
    ],  # all dials where a constraint is repeated more than a number of times configured in the evaluator configuration
    [
        "igcd_f1",
        "informed_domains_outside_goals",
    ],  # dials where the user informed constraints from a domain not included in goals
    [
        "rgcd_f1",
        "requests_not_in_goal",
    ],  # dials where the user requested info that was not in the goal
    [
        "rgcd_f1",
        "requested_domains_outside_goals",
    ],  # dials where the user requested info from domains outside the goal
    [
        "rgcd_f1",
        "provision_entire_domain",
    ],  # dials where the user did not request any info but the sys provided all the information
    [
        "rgcd_f1",
        "provision_slots",
    ],  # dials where the user requested some info and the system provided some of the information
    [
        "rgcd_f1",
        "auto_ref",
    ],  # dials where the user requested a slot that is not in goal but is not penalised (e.g., ref slot never in goal but user should always request it in MultiWOZ)
    [
        "rgcd_f1",
        "multiple_req",
    ],  # dials where the user requested the same information multiple times
    [
        "rgcd_f1",
        "multiple_slot_requests",
        "delayed_response",
    ],  # dials where the user repeated requests because the sys does not immediately respond
    [
        "rgcd_f1",
        "multiple_slot_requests",
        "user_prempted_request",
    ],  # dials where the user requested the info before giving all attributes and then requested again after answering sys question
    [
        "rgcd_f1",
        "multiple_slot_requests",
        "repeated_despite_receiving_answer",
    ],  # dials where the user repeats requests for no good reason
    [
        "rgcd_f1",
        "multiple_slot_requests",
        "system_req_nlu_failure",
    ],  # dials where the user repeats requests because the sys NLU did not recognise the request
    ["rgcd_f1", "multiple_slot_requests", "spurious_annotation"],  # heuristic, ignore
    ["rgcd_f1", "dial_id"],
    [
        "rgcd_f1",
        "repetitions_warning",
    ],  # dials where a request is repeated more than a configured number of times
)
"""This tuple is a summary of the context-driven features and the various behaviours considered.
Read in conjunction with the gcdf1.metrics module and gcdf1.behaviours.multiwoz module. """


def metadata_store_factory() -> dict:
    """Data store for computed metrics.

    Returns
    --------
    A dictionary which contains variable depth nested dictionaries which can be used to log
    metadata during metrics computation.

    Note
    ----
    If adding a new behaviour, remember to update `metadata_to_agg` so that the evaluator aggregates the metric
    metadata across the entire corpus (i.e., creates a single dictionary where each dialogue is a key at the end of the
    evaluation).
    """
    return {
        "igcd_f1": {
            "preprocessed_user_constraints": nested_defaultdict(list, depth=4),
            "preprocessed_sys_constraints": nested_defaultdict(list, depth=4),
            "sys_preempted_constraint": nested_defaultdict(list, depth=2),
            "preprocessed_sys_nlu": nested_defaultdict(list, depth=4),
            "wrong_taxi_annotations": nested_defaultdict(list, depth=1),
            "single_repetitions_nog": nested_defaultdict(list, depth=2),
            "missed_constraints": nested_defaultdict(list, depth=2),
            "informed_wrong_value": nested_defaultdict(list, depth=2),
            "failed_dialogue": nested_defaultdict(list, depth=2),
            "missed_domain": nested_defaultdict(list, depth=1),
            "multiple_constraint_informs": {
                "elicited_by_sys_question": nested_defaultdict(list, depth=2),
                "failed_sys_nlu": nested_defaultdict(list, depth=2),
                "elicited_by_booking_failure": nested_defaultdict(list, depth=2),
                "elicited_by_no_offers": nested_defaultdict(list, depth=2),
                "nobook_validation_failure": nested_defaultdict(list, depth=2),
                "nooffer_validation_failure": nested_defaultdict(list, depth=2),
                "confirmed_choice_params": nested_defaultdict(list, depth=2),
                "repeated_while_requesting_info": nested_defaultdict(list, depth=2),
                "repeated_while_booking": nested_defaultdict(list, depth=2),
                "repeated_when_answering_request": nested_defaultdict(list, depth=2),
                "switched_between_domains": nested_defaultdict(list, depth=2),
                "unmatched": nested_defaultdict(list, depth=2),
                "wrong_taxi_annotations": nested_defaultdict(list, depth=2),
            },
            "constraints_not_in_goal": {
                "auto_matched_slot": nested_defaultdict(list, depth=2),
                "auto_matched_value": nested_defaultdict(list, depth=2),
                "auto_matched_slot_value_pair": nested_defaultdict(list, depth=2),
                "first_turn": nested_defaultdict(list, depth=2),
                "elicited_by_sys_question": nested_defaultdict(list, depth=2),
                "elicited_by_sys_offer": nested_defaultdict(list, depth=2),
                "elicited_by_booking_failure": nested_defaultdict(list, depth=2),
                "elicited_by_booking_acceptance": nested_defaultdict(list, depth=2),
                "elicited_by_recommendation": nested_defaultdict(list, depth=2),
                "unmatched": nested_defaultdict(list, depth=2),
            },
            "slots_missing_from_cmap": nested_defaultdict(list, depth=3),
            "dial_id": nested_defaultdict(list, depth=1),
            "repeated_inform": nested_defaultdict(list, depth=2),
            "informed_domains_outside_goals": nested_defaultdict(list, depth=2),
            "repetitions_warning": nested_defaultdict(list, depth=2),
        },
        "rgcd_f1": {
            "dial_id": nested_defaultdict(list, depth=1),
            "multiple_slot_requests": {
                "system_req_nlu_failure": nested_defaultdict(list, depth=2),
                "repeated_despite_receiving_answer": nested_defaultdict(list, depth=2),
                "user_prempted_request": nested_defaultdict(list, depth=2),
                "delayed_response": nested_defaultdict(list, depth=2),
                "spurious_annotation": nested_defaultdict(list, depth=2),
            },
            "requests_not_in_goal": nested_defaultdict(list, depth=2),
            "requested_domains_outside_goals": nested_defaultdict(list, depth=2),
            "provision_entire_domain": nested_defaultdict(list, depth=2),
            "provision_slots": nested_defaultdict(list, depth=2),
            "auto_ref": defaultdict(list),
            "multiple_req": nested_defaultdict(list, depth=2),
            "repetitions_warning": nested_defaultdict(list, depth=2),
        },
    }

from __future__ import annotations

from collections import defaultdict
from contextlib import suppress

from omegaconf import DictConfig

import gcdf1.multiwoz_metadata as metadata
from gcdf1.utils.dialogue import (
    ACT_SLOT_SEP,
    SLOT_VALUE_SEP,
    get_turn_action_params,
    get_turn_actions,
)
from gcdf1.utils.utils import dispatch_on_value


def has_understood_request(
    sys_nlu: dict, slot: str, domain: str, lowercase_slots: bool = True
) -> bool:
    """Check if the system has understood a user request in a particular domain."""

    # assume perfect system if NLU not available
    if not sys_nlu:
        return True

    sys_nlu_requested = get_turn_action_params(
        sys_nlu,
        act_patterns=metadata.REQUEST_ACT_PATTERNS,
        service_patterns=[domain],
        include_values=False,
        use_lowercase=lowercase_slots,
    )[
        domain
    ]  # type: list[str]

    assert all("-" not in slt for slt in sys_nlu_requested)
    sys_nlu_requested = [f"{domain}-{slt}" for slt in sys_nlu_requested]

    return slot in sys_nlu_requested


def confirms_booking(domain: str, sys_turn: dict, use_lowercase: bool = True) -> bool:
    """Check if the system turn contains a booking reference."""

    if domain == "train":
        patterns = metadata.TRAIN_CONFIRMATION_PATTERNS + metadata.INFORM_ACT_PATTERNS
        sys_informed_slots = get_turn_action_params(
            sys_turn,
            act_patterns=patterns,
            service_patterns=[domain],
            include_values=False,
            use_lowercase=use_lowercase,
        )[domain]
        if set(sys_informed_slots).intersection(set(metadata.REF_SLOT_PATTERNS)):
            return True
    else:
        sys_info_slots_by_service = get_turn_action_params(
            sys_turn,
            service_patterns=[domain, "booking"],
            act_patterns=metadata.BOOK_ACT_PATTERNS + metadata.INFORM_ACT_PATTERNS,
            include_values=False,
            use_lowercase=use_lowercase,
        )
        sys_informed_slots = []
        for domain in [domain, "booking"]:
            sys_informed_slots += sys_info_slots_by_service[domain]
        return bool(set(sys_informed_slots).intersection(metadata.REF_SLOT_PATTERNS))
    return False


def is_ref_slot(slot: str) -> bool:
    """Check if a given slot is a reference number slot."""
    return any(substr in slot for substr in metadata.REF_SLOT_PATTERNS)


def has_given_answer(
    sys_turn: dict, slot: str, domain: str, use_lowercase: bool = True
) -> bool:

    # special handling for `ref` slot as it can be informed as part
    # of 'book' acts (hotel, restaurant) but also 'offerbooked' act (train)
    if is_ref_slot(slot):
        if confirms_booking(domain, sys_turn, use_lowercase=use_lowercase):
            return True

    sys_informed = get_turn_action_params(
        sys_turn,
        act_patterns=metadata.INFORM_ACT_PATTERNS,
        service_patterns=[domain],
        include_values=True,
        use_lowercase=use_lowercase,
    )[
        domain
    ]  # type: list[str]

    sys_informed_params = [
        action_param.split(SLOT_VALUE_SEP) for action_param in sys_informed
    ]
    if not sys_informed_params:
        return False
    # TODO: RESTRICTING TO MULTIWOZ 2.1!!
    sys_informed_slots = [
        f"{domain}-{action_param_lst[0]}" for action_param_lst in sys_informed_params
    ]
    sys_informed_vals = [
        action_param_lst[1] for action_param_lst in sys_informed_params
    ]
    assert len(sys_informed_vals) == len(sys_informed_slots)
    if slot not in sys_informed_slots:
        return False
    else:
        slot_idx = sys_informed_slots.index(slot)
        value = sys_informed_vals[slot_idx]
        assert value
        return True


def is_system_triggered_repeated_request(
    sys_turns: list[dict], slot: str, domain: str, config: DictConfig
) -> list[str]:
    """Checks if a requested slot repetition is due to a system behaviour such as:

        - system not understanding the user request

        - system policy not immediately responding to the user request (e.g., due to error \
        or because some constraints are missing or the user has not yet confirmed they want \
        to go ahead with a transaction)

    Parameters
    ----------
    slot
        Slot requested multiple times.
    domain
        The domain in which the request was made.
    sys_turns
        The span of system turns in between a repeated request for `slot`.
    config
        See configs/multiwoz_user_evaluator.yaml, rgcd_f1 key, for configuration documentation.
    """

    behaviours = []
    if not config.validate_nlu and not config.validate_policy:
        return behaviours
    if config.validate_nlu:
        turn = sys_turns[-1]
        understood_req = has_understood_request(
            turn.get("nlu", {}),
            slot,
            domain,
            lowercase_slots=config.use_lowercase,
        )
        if not understood_req:
            behaviours.append("system_req_nlu_failure")

    if config.validate_policy:
        system_responses, system_info_requests = [], []
        assert sys_turns
        for turn in reversed(sys_turns):
            system_responded = has_given_answer(
                turn, slot, domain, use_lowercase=config.use_lowercase
            )
            sys_asked_more_info = information_requested(
                domain, turn, use_lowercase=config.use_lowercase
            )
            system_responses.append(system_responded)
            system_info_requests.append(sys_asked_more_info)

        if any(system_responses):
            behaviours.append("repeated_despite_receiving_answer")
        else:
            if system_info_requests[0]:
                behaviours.append("user_prempted_request")
            else:
                behaviours.append("delayed_response")
    return behaviours


def system_requested_slot(
    domain: str, slot: str, sys_turn: dict, use_lowercase: bool = True
) -> bool:
    """Returns `True` if the system requested the given slot in the given user turn."""

    sys_requested_slots_by_service = get_turn_action_params(
        sys_turn,
        act_patterns=metadata.REQUEST_ACT_PATTERNS,
        service_patterns=[domain, "booking"],
        include_values=False,
        use_lowercase=use_lowercase,
    )  # type: defaultdict[str, list[str]]

    assert "-" not in slot
    assert all(("-" not in slot for slot in sys_requested_slots_by_service[domain]))
    assert all(("-" not in slot for slot in sys_requested_slots_by_service["booking"]))
    return (
        slot in sys_requested_slots_by_service[domain]
        or slot in sys_requested_slots_by_service["booking"]
    )


def system_offered_slot(
    domain: str,
    slot: str,
    sys_turns: list[dict],
    use_lowercase: bool = True,
    entity_slots: bool = False,
) -> bool:
    """Returns `True` if the system informed the slot in any of the given turns."""

    # TODO: THIS CAN BE IMPROVED BY ALSO CHECKING THE VALUES
    assert "-" not in slot
    if use_lowercase:
        slot = slot.lower()
    if entity_slots:
        if slot not in metadata.ENTITY_SLOTS_BY_SERVICE[domain]:
            return False
    offered_slots = False
    patterns = (
        metadata.INFORM_ACT_PATTERNS
        + metadata.RECOMMENDATION_PATTERNS
        + metadata.BOOK_ACT_PATTERNS
    )
    for turn in sys_turns:
        turn_action_params = get_turn_action_params(
            turn,
            act_patterns=patterns,
            service_patterns=[domain] + metadata.BOOKING_DOMAIN_PATTERNS,
            include_values=False,
            use_lowercase=use_lowercase,
        )
        turn_action_params = turn_action_params[domain] + turn_action_params["booking"]
        if any(slot in param for param in turn_action_params):
            assert all("-" not in par for par in turn_action_params)
            offered_slots = True
            break
    return offered_slots


def system_elicited_new_constraint(
    domain: str, constraint: str, sys_turns: list[dict], config
) -> list[str]:
    """Returns `True` if the given constraint, not present in the goal, has been informed by the user as a
    result of a specific system behaviour.

    # TODO: BULLET POINTS WITH BEHAVIOURS CONSIDERED
    """

    behaviours = []
    context_size = config.sys_context_size
    slot, value = constraint.split(SLOT_VALUE_SEP)
    if config.requests:
        # only previous turns considered
        sys_turn = sys_turns[0]
        system_asked_question = system_requested_slot(
            domain, slot, sys_turn, use_lowercase=config.use_lowercase
        )
        if system_asked_question:
            behaviours.append("elicited_by_sys_question")
    if config.offers:
        system_offered = system_offered_slot(
            domain, slot, sys_turns[:context_size], use_lowercase=config.use_lowercase
        )
        if system_offered:
            behaviours.append("elicited_by_sys_offer")
    if config.booking_failure_response:
        # only check if `nobook<name` act was generated by setting entities_only arg
        selected_new_entity = notifies_booking_failure(
            domain,
            slot,
            sys_turns[:context_size],
            entities_only=True,
            use_lowercase=config.use_lowercase,
        )
        if selected_new_entity:
            behaviours.append("elicited_by_booking_failure")
    if config.booking_request_response:
        reacts_to_booking_intent = offers_booking_intent(
            domain,
            slot,
            sys_turns[:1],
            use_lowercase=config.use_lowercase,
        )
        if reacts_to_booking_intent:
            behaviours.append("elicited_by_booking_acceptance")
    if config.recommendations:
        # consider all previous sys turns for recommendations
        # useful when turkers forget to ask requests (e.g, MUL0828)
        recommendation_response = makes_recommendation(
            domain,
            slot,
            sys_turns,
            use_lowercase=config.use_lowercase,
        )
        if recommendation_response:
            behaviours.append("elicited_by_recommendation")
    return behaviours


def makes_recommendation(
    domain: str,
    slot: str,
    sys_turns: list[dict],
    use_lowercase: bool = True,
    entity_slots: bool = True,
):
    if use_lowercase:
        slot = slot.lower()
    # this flag deals with attraction-recommend<area=east and similar
    # annotations
    if entity_slots:
        if slot not in metadata.ENTITY_SLOTS_BY_SERVICE[domain]:
            return False
    for sys_turn in sys_turns:
        sys_actions = get_turn_actions(
            sys_turn,
            act_patterns=metadata.RECOMMENDATION_PATTERNS,
            use_lowercase=use_lowercase,
        )[domain]
        if sys_actions:
            return True


@dispatch_on_value
def offers_booking_intent(
    domain: str,
    slot: str,
    sys_turns: list[dict],
    use_lowercase: bool = True,
    entities_only: bool = True,
):
    """Checks if a booking intent was offered in the first turn."""

    if use_lowercase:
        slot = slot.lower()
    if domain not in metadata.TRANSACTIONAL_DOMAINS:
        return False
    if entities_only:
        if slot not in metadata.ENTITY_SLOTS_BY_SERVICE[domain]:
            return False
    sys_turn = sys_turns[0]
    # TODO: THIS WILL NOT WORK IF BOOKING-INFORM IS NOT IN THE SCHEMA
    sys_actions = get_turn_actions(
        sys_turn,
        act_patterns=metadata.BOOKING_REQUEST_PATTERNS,
        use_lowercase=use_lowercase,
    )
    if sys_actions:
        assert (
            "booking" in sys_actions
            or "Booking" in sys_actions
            or domain in sys_actions
        )
        return True


@offers_booking_intent.register("train")
def _(
    domain: str,
    slot: str,
    sys_turns: list[dict],
    use_lowercase: bool = True,
    entities_only: bool = True,
) -> bool:
    """Detects if the system offers a booking intent for the train domain in any of the system turns."""
    if entities_only:
        if slot not in metadata.ENTITY_SLOTS_BY_SERVICE[domain]:
            return False

    for turn in sys_turns:
        actions = get_turn_actions(
            turn,
            act_patterns=metadata.TRAIN_BOOKING_INTENT_PATTERNS,
            service_patterns=[domain],
            use_lowercase=use_lowercase,
        )[domain]
        acts = [a.split(ACT_SLOT_SEP)[0] for a in actions]
        for act in acts:
            if not any(act == patt for patt in metadata.TRAIN_CONFIRMATION_PATTERNS):
                return True
    return False


def notifies_booking_failure(
    domain: str,
    slot: str,
    sys_turns: list[dict],
    entities_only: bool = True,
    use_lowercase: bool = True,
) -> bool:
    """Checks if any of the system turns inform the user of a booking failure.

    Returns
    -------
    A boolean which indicates whether a booking failure pattern was found in one of the turns.
    The function returns as soon as if finds the pattern for the first time.
    """

    if domain not in metadata.TRANSACTIONAL_DOMAINS:
        return False
    if entities_only:
        if slot not in metadata.ENTITY_SLOTS_BY_SERVICE[domain]:
            return False
    for turn in sys_turns:
        sys_actions = get_turn_actions(
            turn,
            act_patterns=metadata.TRANSACTION_FAILURE_ACT_PATTERNS,
            use_lowercase=use_lowercase,
        )
        if sys_actions:
            assert (
                "booking" in sys_actions
                or "Booking" in sys_actions
                or domain in sys_actions
            )
            return True
    return False


def notifies_void_result(
    domain: str, sys_turn: dict, use_lowercase: bool = True
) -> bool:

    sys_actions_by_service = get_turn_actions(
        sys_turn,
        act_patterns=metadata.NOTIFY_FAILURE_PATTERNS,
        use_lowercase=use_lowercase,
    )
    assert not ("booking" in sys_actions_by_service)
    if sys_actions_by_service[domain]:
        return True
    return False


def has_understood_constraint(
    domain: str, constraint: str, sys_nlu_turns: dict
) -> bool:
    return constraint in sys_nlu_turns.get(domain, {})


def is_system_elicited_repetition(
    domain: str,
    constraint: str,
    sys_turns: list[dict],
    config,
    sys_nlu: dict[str, dict[str, list[int]]],
) -> list[str]:

    behaviours = []

    slot, value = constraint.split(SLOT_VALUE_SEP)
    if config.requests:
        sys_turn = sys_turns[0]
        sys_asked_question = system_requested_slot(
            domain, slot, sys_turn, use_lowercase=config.use_lowercase
        )
        if sys_asked_question:
            behaviours.append("elicited_by_sys_question")
    if config.booking_failure_response:
        failed_booking_turns, booking_made = [], []
        for idx, turn in enumerate(sys_turns):
            sys_informed_booking_failure = notifies_booking_failure(
                domain,
                slot,
                [turn],
                entities_only=False,
                use_lowercase=config.use_lowercase,
            )
            if sys_informed_booking_failure:
                failed_booking_turns.append(idx)
            # if, most recently, the booking is a successful (aka we have ref number),
            # then the user has other reason for repetition!
            sys_informed_booking_success = confirms_booking(
                domain, turn, use_lowercase=config.use_lowercase
            )
            # and so do they if an offer was made or another entity is discussed
            sys_made_offer_or_proposed_entity = _offers_entity_or_booking(
                domain, [turn]
            )
            if sys_informed_booking_success or sys_made_offer_or_proposed_entity:
                booking_made.append(idx)

        if failed_booking_turns:
            if not booking_made:
                behaviours.append("elicited_by_booking_failure")
            else:
                # most recent notification is failure (see SNG0611 T7 to understand equality case)
                if failed_booking_turns[0] <= booking_made[0]:
                    behaviours.append("elicited_by_booking_failure")
                else:
                    behaviours.append("nobook_validation_failure")
    if config.no_offers:
        noofer_turns, booking_offered_or_made = [], []
        for idx, turn in enumerate(sys_turns):
            sys_informed_no_results = notifies_void_result(
                domain, turn, use_lowercase=config.use_lowercase
            )
            if sys_informed_no_results:
                noofer_turns.append(idx)

            sys_informed_booking_success = confirms_booking(
                domain, turn, use_lowercase=config.use_lowercase
            )
            # if this happens, the discussion might have moved on already
            sys_made_offer_or_proposed_entity = _offers_entity_or_booking(
                domain, [turn]
            )
            if sys_informed_booking_success or sys_made_offer_or_proposed_entity:
                booking_offered_or_made.append(idx)
        if noofer_turns:
            if not booking_offered_or_made:
                behaviours.append("elicited_by_no_offers")
            else:
                # no-offer and booking-inform can be informed in the same turn (PMUL0276, T3)
                if noofer_turns[0] <= booking_offered_or_made[0]:
                    behaviours.append("elicited_by_no_offers")
                else:
                    behaviours.append("nooffer_validation_failure")
    if config.nlu:
        understood_constr = has_understood_constraint(domain, constraint, sys_nlu)
        if not understood_constr:
            behaviours.append("failed_sys_nlu")

    return behaviours


def system_offered_choice(domain: str, sys_turn: dict, use_lowercase: bool = True):
    """Checks if the system offered the user a choice of venues."""
    sys_actions_by_service = get_turn_action_params(
        sys_turn, include_values=False, use_lowercase=use_lowercase
    )
    sys_actions = sys_actions_by_service[domain]
    if sys_actions:
        return any(
            pattern in sys_actions_by_service[domain]
            for pattern in metadata.CHOICE_PATTERNS
        )
    return False


def information_requested(domain: str, turn: dict, use_lowercase: bool = True) -> bool:
    """Checks if request in a given service was made."""

    req_actions = get_turn_actions(
        turn,
        act_patterns=metadata.REQUEST_ACT_PATTERNS,
        service_patterns=[domain] + metadata.BOOKING_DOMAIN_PATTERNS,
        use_lowercase=use_lowercase,
    )
    if req_actions[domain] or req_actions["booking"]:
        return True
    return False


@dispatch_on_value
def _offers_entity_or_booking(domain: str, sys_turns: list[dict]) -> bool:
    """Checks if any of the turns an entity is proposed by the system via an inform-name,
    recommend-name, select-name act or if there is a booking intent (via booking-inform
    """

    # TODO: SHOULD THIS FUNCTION ALSO CONSIDER THE CHOICE SLOT?

    entity_slots = set(slt.lower() for slt in metadata.ENTITY_SLOTS_BY_SERVICE[domain])
    if not entity_slots:
        return False

    # only the system turn before considered throughout
    for turn in sys_turns:
        recommended, offered_entity = [], []
        for slot in entity_slots:
            sys_recommended = makes_recommendation(
                domain,
                slot,
                [turn],
                use_lowercase=True,
            )
            # checks also inform act, above only recommend/select
            sys_offered_entity = system_offered_slot(
                domain,
                slot,
                [turn],
                use_lowercase=True,
            )
            recommended.append(sys_recommended)
            offered_entity.append(sys_offered_entity)

        # condition is that all entity slots have been offered or recommended
        # in multiwoz, there is just one slot (name/trainID) but SGD uses multiple
        # slots to refer to an entity.
        if not recommended and not offered_entity:
            return False
        if all(recommended) or all(offered_entity):
            return True

        # this just finds `booking-inform`
        booking_intent_offered = offers_booking_intent(
            domain,
            "name",  # slot name does not matter because we just check if we have booking inform
            [turn],
            use_lowercase=True,
            entities_only=False,
        )
        # and this ensures that it's not a booking failure, we just look for 'nobook' act,
        # slot name does not matter
        booking_failure_notified = notifies_booking_failure(
            domain, "name", [turn], entities_only=False, use_lowercase=True
        )
        if booking_intent_offered and not booking_failure_notified:
            return True
    return False


@_offers_entity_or_booking.register("train")
def _(domain: str, sys_turns: list[dict]) -> bool:

    entity_slots = set(slt.lower() for slt in metadata.ENTITY_SLOTS_BY_SERVICE[domain])

    for turn in sys_turns:
        turn_slots = get_turn_action_params(
            turn,
            act_patterns=metadata.INFORM_ACT_PATTERNS,
            service_patterns=[domain],
            include_values=False,
            use_lowercase=True,
        )[domain]
        if entity_slots.issubset(set(turn_slots)):
            return True
        turn_actions = get_turn_actions(
            turn,
            act_patterns=metadata.TRAIN_BOOKING_INTENT_PATTERNS,
            use_lowercase=True,
        )
        if turn_actions:
            assert "train" in turn_actions
            return True
    return False


def detect_domain_switches(domain: str, usr_turn: dict, sys_turns: list[dict]) -> bool:
    """Detects if other domains than the current domain were discussed in the system turns."""

    usr_actions_by_service = get_turn_actions(usr_turn, service_patterns=["taxi"])
    # likely an annotation error
    if usr_actions_by_service or "taxi" in usr_turn["utterance"]:
        return False
    domains_discussed = set()
    for turn in sys_turns:
        actions_by_service = get_turn_actions(turn)
        if "taxi" in actions_by_service or "taxi" in turn["utterance"]:
            return False
        with suppress(KeyError):
            actions_by_service.pop("booking")
        domains_discussed.update(actions_by_service.keys())
    if domain in domains_discussed:
        domains_discussed.remove(domain)
    if domains_discussed:
        return True
    return False


def is_behavioural_repetition(
    domain: str, constraint: str, usr_turn: dict, sys_turns: list[dict], config
) -> list[str]:
    """Checks if a repetition is due to a specific user behaviour.

    Returns
    -------
    behaviours
        A list with the rules that apply to the repetition of the current constraint.

    """
    behaviours = []
    slot, value = constraint.split(SLOT_VALUE_SEP)
    if config.offer_checks:
        # consider only previous turn for this pattern
        sys_turn = sys_turns[0]
        system_offed_choice = system_offered_choice(
            domain, sys_turn, use_lowercase=config.use_lowercase
        )
        if system_offed_choice:
            behaviours.append("confirmed_choice_params")
    if config.repeats_while_requesting_info:
        repeated_while_requesting = information_requested(
            domain, usr_turn, use_lowercase=config.use_lowercase
        )
        if repeated_while_requesting:
            behaviours.append("repeated_while_requesting_info")
    if config.repeats_while_booking:
        repeated_while_booking = _offers_entity_or_booking(domain, [sys_turns[0]])

        if repeated_while_booking:
            if slot in metadata.TRAIN_BOOKING_INTENT_PATTERNS and taxi_active_intent(
                usr_turn
            ):
                return ["wrong_taxi_annotations"]
            behaviours.append("repeated_while_booking")
    if config.repeats_when_answering_request:
        sys_requested = information_requested(
            domain,
            sys_turns[0],
            use_lowercase=config.use_lowercase,
        )
        if sys_requested:
            behaviours.append("repeated_when_answering_request")
    if config.proposes_multiple_domains:
        domain_switched = detect_domain_switches(domain, usr_turn, sys_turns)
        if domain_switched:
            behaviours.append("switched_between_domains")
    return behaviours


def taxi_active_intent(turn: dict) -> bool:
    for frame in turn["frames"]:
        if frame["service"] == "taxi" and frame["actions"]:
            return True
    return False

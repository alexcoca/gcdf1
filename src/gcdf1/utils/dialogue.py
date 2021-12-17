"""A module which contains utility function for inspecting and extracting
data from dialogues and training splits. Use in conjunction with the
information in the `metadata` module.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Literal, Optional

from gcdf1.utils.data import actions_iterator, dialogue_iterator
from gcdf1.utils.utils import append_to_values

MULTIPLE_VALUE_SEP = "###"
SLOT_VALUE_SEP = "==="
ACT_SLOT_SEP = "<<<"
SYSTEM_AGENT = "SYSTEM"
USER_AGENT = "USER"


def has_requestables(dialogue: dict) -> bool:
    """Returns `True` if the user requests information
    from the system and false otherwise.
    """

    for turn in dialogue_iterator(dialogue, user=True, system=False):
        for frame in turn["frames"]:
            if frame["state"]["requested_slots"]:
                return True
    return False


def get_intents(dialogue: dict, exclude_none: bool = True) -> set[str]:
    """Returns the intents in a dialogue.

    Parameters
    ----------
    dialogue
        Nested dictionary containing dialogue and annotations.
    exclude_none
        If True, the `NONE` intent is not included in the intents set.

    Returns
    -------
    intents
        A set of intents contained in the dialogue.
    """

    intents = set()
    for turn in dialogue_iterator(dialogue, user=True, system=False):
        for frame in turn["frames"]:
            intent = frame["state"]["active_intent"]
            if exclude_none:
                if intent == "NONE":
                    continue
                else:
                    intents.add(intent)
            else:
                intents.add(intent)
    return intents


def get_domain_name(service_name: str) -> str:
    """
    Returns the domain name given a service name.

    Params
    -------
    service_name
        Name of the service

    Returns
    -------
    Name of the domain.
    """
    return service_name.split("_")[0]


def get_requestables_set(
    dialogue: dict, agent: Literal["USER", "SYSTEM"] = "USER"
) -> set[str]:
    """Get requestable slots in a given dialogue.

    Parameters
    ----------
    dialogue
        A nested dictionary representation of the dialogue in SGD format.
    agent
        The agent whose requested slots are returned.

    Returns
    -------
    requestables
        set of slots which appear in the ``['state']['requested_slots']`` of all frames for the specified agent.
    """

    user_flag = True if agent == "USER" else False
    sys_flag = True if agent == "SYSTEM" else False

    requestables = set()
    for turn in dialogue_iterator(dialogue, user=user_flag, system=sys_flag):
        for frame in turn["frames"]:
            if reqs := frame["state"]["requested_slots"]:
                requestables.update(reqs)
    return requestables


def get_turn_services(turn: dict) -> list[str]:
    return [frame["service"] for frame in turn["frames"]]


def get_turn_idx(turn_idx: int, user_flag: bool, sys_flag: bool) -> int:
    """Correct the turn index when excluding one of the speakers from dialogue iteration."""

    # TODO: MAKE ARGS 2,3 REQUIRED KWARGS TO IMPROVE READABILITY
    # TODO: TEST THIS FUNCTION
    # TODO: LONG TERM: REFACTOR SO THE DIALOGUE ITERATOR RETURNS THE CORRECT ID
    if user_flag != sys_flag:
        if sys_flag:
            turn_idx = turn_idx * 2 + 1
        if user_flag:
            turn_idx *= 2
    return turn_idx


def get_requestables_by_service(
    dialogue: dict,
    agent: Literal["USER", "SYSTEM"] = "USER",
    use_lowercase: bool = True,
) -> dict[str, dict[str, list[int]]]:
    """Get a mapping from services to slot names requested for that service. If a slot is requested multiple times,
    it will appear multiple times in the value lists.

    Parameters
    ----------
    dialogue
        Dialogue from which requestable slots are to be extracted, in SGD format.
    agent
        A string that should be either 'USER' or 'SYSTEM' to indicate which turns to extract requested slots from.
    use_lowercase
        Lowercases slots before retuning them.


    Returns
    -------
    requested_slots
        A mapping of the form::

            {
            'domain_name': {'slot_name': list[int]}
            }

        where each slot name is mapped to a list of turn ids where it is requested.
    """

    user_flag = True if agent == "USER" else False
    sys_flag = True if agent == "SYSTEM" else False

    requested_slots = defaultdict(lambda: defaultdict(list))
    for turn_idx, turn in enumerate(
        dialogue_iterator(dialogue, system=sys_flag, user=user_flag)
    ):
        turn_idx = get_turn_idx(turn_idx, user_flag, sys_flag)
        assert turn_idx == int(turn["turn_id"])
        this_turn_req_slots = defaultdict(dict)
        for frame in turn["frames"]:
            if reqt := frame["state"]["requested_slots"]:
                for slot in reqt:
                    if use_lowercase:
                        slot = slot.lower()
                    this_turn_req_slots[frame["service"]][slot] = [turn_idx]
        append_to_values(requested_slots, this_turn_req_slots)

    return requested_slots


def get_turn_requestables_by_service(turn: dict) -> dict[str, list]:
    """Retrieve the requested slots in a turn, broken down by service.

    Parameters:
    ----------
    turn
        Turn, in SGD format.

    Returns
    -------
    A mapping of the form::

        {
        'service_name': list[str],
        }

    where the lists contain service names.
    """

    this_turn_reqt_slots = defaultdict(list)
    for frame in turn["frames"]:
        if reqt := frame["state"]["requested_slots"]:
            this_turn_reqt_slots[frame["service"]].extend(reqt)

    return this_turn_reqt_slots


def get_slots(
    dialogue: dict,
    agent: Literal["USER", "SYSTEM"],
    act_patterns: Optional[list[str]] = None,
    service_patterns: Optional[list[str]] = None,
    use_lowercase: bool = True,
) -> dict[str, dict[str, list[int]]]:
    """Retrieve slots from dialogue turns, by service."""

    # TODO: TEST THIS FUNCTION

    user_flag = True if agent in ["USER", "BOTH"] else False
    sys_flag = True if agent in ["SYSTEM", "BOTH"] else False
    slots_by_service = defaultdict(lambda: defaultdict(list))
    for turn_idx, turn in enumerate(
        dialogue_iterator(dialogue, user=user_flag, system=sys_flag)
    ):
        this_turn_slots = defaultdict(dict)
        turn_idx = get_turn_idx(turn_idx, user_flag, sys_flag)
        this_turn_slots_by_service = get_turn_action_params(
            turn,
            act_patterns=act_patterns,
            service_patterns=service_patterns,
            include_values=False,
            use_lowercase=use_lowercase,
        )  # type: dict[str, list[str]]

        for service in this_turn_slots_by_service:
            for slot in this_turn_slots_by_service[service]:
                this_turn_slots[service][slot] = [turn_idx]
        if this_turn_slots:
            append_to_values(slots_by_service, this_turn_slots)
    return slots_by_service


def get_entity_slots(dialogue: dict):
    raise NotImplementedError


def get_utterances(dialogue: dict) -> list[str]:
    """
    Retrieves all utterances from a dialogue.

        See `get_dialogue_outline` for structure.

    Returns
    -------
        Utterances in the input dialogue.
    """
    return [f'{turn["speaker"]} {turn["utterance"]}' for turn in dialogue["turns"]]


def get_turn_actions(
    turn: dict,
    act_patterns: Optional[list[str]] = None,
    service_patterns: Optional[list[str]] = None,
    use_lowercase: bool = True,
    slot_value_sep: str = SLOT_VALUE_SEP,
    act_slot_sep: str = ACT_SLOT_SEP,
    multiple_val_sep: str = MULTIPLE_VALUE_SEP,
) -> dict[str, list[str]]:
    """
    Retrieve actions from a given dialogue turn. An action is a parametrised dialogue act (e.g., INFORM(price=cheap)).

    Parameters
    ----------
    turn
        Contains turn and annotations, with the structure::

            {
            'frames': [
                    {
                        'actions': dict,
                        'service': str,
                        'slots': list[dict], can be empty if no slots are mentioned (e.g., "I want to eat.") , in SYS \
                                 turns or if the USER requests a slot (e.g., address). The latter is tracked in the
                                 ``'state'`` dict.
                        'state': dict
                    },
                    ...
                ],
            'speaker': 'USER' or 'SYSTEM',
            'utterance': str,

            }

        The ``'actions'`` dictionary has structure::

            {
            'act': str (name of the act, e.g., INFORM_INTENT(intent=findRestaurant), REQUEST(slot))
            'canonical_values': [str] (name of the acts). It can be the same as value for non-categorical slots. Empty
                for some acts (e.g., GOODBYE)
            'slot': str, (name of the slot that parametrizes the action, e.g., 'city'. Can be "" (e.g., GOODBYE())
            'values': [str], (value of the slot, e.g "San Jose"). Empty for some acts (e.g., GOODBYE()), or if the user
                makes a request (e.g., REQUEST('street_address'))
            }

        When the user has specified all the constraints (e.g., restaurant type and location), the next ``'SYSTEM'`` turn
        has the following _additional_ keys of the ``'actions'`` dictionary:

            {
            'service_call': {'method': str, same as the intent, 'parameters': {slot:value} specified by user}
            'service_result': [dict[str, str], ...] where each dict maps properties of the entity retrieved to their
                vals. Structure depends on the entity retrieved.
            }

        The dicts of the ``'slots'`` list have structure:

            {
            'exclusive_end': int (char in ``turn['utterance']`` where the slot value ends)
            'slot': str, name of the slot
            'start': int (char in ``turn['utterance']`` where the slot value starts)
            }

        The ``'state'`` dictionary has the structure::

            {
            'active_intent': str, name of the intent active at the current turn,
            'requested_slots': [str], slots the user requested in the current turn
            'slot_values': dict['str', list[str]], mapping of slots to values specified by USER up to current turn
            }

    act_patterns
        Optionally specify these patterns to return only specific actions. The patterns are matched against
        ``turn['frames'][frame_idx]['actions'][action_idx]['act'] for all frames and actions using ``re.search``.
    service_patterns
        Optionally specify these patterns to return only actions from certain services. The patterns are matched against
        ``turn['frames'][frame_idx]['service']``.
    use_lowercase
        Lowercase the action strings before returning.
    slot_value_sep, act_slot_sep, multiple_val_sep
        Separators used to format the action strings.

    Returns
    -------
    formatted_actions
        Actions in the current dialogue turn. The format is::

            {
                'service_name': list[str]
            }

        where each string is an action formatted as act{act_slot_sep}slot{slot_value_sep}{val_1}{multiple_val_sep}val_2.
        For example, INFORM<<<price===cheap assuming there is a single value, slot_value_sep is === and
        act_slot_sep is <<<.
    """

    # TODO: TEST THIS FUNCTION, VERY IMPORTANT => can a string end in slot value sep?

    formatted_actions = defaultdict(list)

    for frame in turn.get("frames", []):
        service = frame.get("service", "")
        # return patterns only for certain services
        if service_patterns:
            if not any((re.search(pattern, service) for pattern in service_patterns)):
                continue
        for action_dict in actions_iterator(frame, patterns=act_patterns):
            # empty frame
            if action_dict is None:
                continue
            # acts without parameters (e.g., goodbye)
            slot = ""
            if "slot" in action_dict:
                slot = action_dict["slot"] if action_dict["slot"] else ""
            val = ""
            if slot:
                val = (
                    f"{multiple_val_sep}".join(action_dict["values"])
                    if action_dict["values"]
                    else ""
                )

            if slot and val:
                f_action = (
                    f"{action_dict['act']}{act_slot_sep}{slot}{slot_value_sep}{val}"
                )
            else:
                if slot:
                    f_action = f"{action_dict['act']}{act_slot_sep}{slot}"
                else:
                    f_action = f"{action_dict['act']}"
            f_action = f_action.lower() if use_lowercase else f_action
            formatted_actions[service].append(f_action)

    return formatted_actions


def get_params(
    actions: list[str],
    include_values: bool = True,
    slot_value_sep: str = SLOT_VALUE_SEP,
    act_slot_sep: str = ACT_SLOT_SEP,
    use_lowercase: bool = True,
) -> list[str]:
    """Retrieve action parameters given a formatted action.

    Parameters
    ----------
    actions
        Formatted action strings. See get_turn_actions for actions formatting.
    include_values
        Parameters returned include values. Otherwise, value is dropped.
    slot_value_sep, act_slot_sep
        Separators used to mark actions
    use_lowercase
        Lowercase the parameters before returning.

    Returns
    -------
    action_params
        Action parameters extracted from input actions. This may not be the same len as input
        list if some of the input actions do not have any parameters.
    """

    action_params = []
    for f_action in actions:
        this_action_params = "".join(f_action.split(act_slot_sep)[1:])
        if this_action_params:
            if use_lowercase:
                action_params.append(this_action_params.lower())
            else:
                action_params.append(this_action_params)
    if not include_values:
        action_params = [
            "".join(action_param.split(slot_value_sep)[0])
            for action_param in action_params
        ]
    return action_params


def get_turn_action_params(
    turn: dict,
    act_patterns: Optional[list[str]] = None,
    service_patterns: Optional[list[str]] = None,
    include_values: bool = True,
    use_lowercase: bool = True,
    slot_value_sep: str = SLOT_VALUE_SEP,
    multiple_val_sep: str = MULTIPLE_VALUE_SEP,
) -> defaultdict[str, list[str]]:
    """Obtain the parameters for all the actions in the turn.

    Parameters
    ---------
        See get turn actions for parameter definitions.

    Returns
    -------
    turn_action_params
        A mapping from services to lists of parameters for the actions in that service.
    """

    turn_actions_by_service = get_turn_actions(
        turn,
        act_patterns=act_patterns,
        service_patterns=service_patterns,
        use_lowercase=use_lowercase,
        slot_value_sep=slot_value_sep,
        multiple_val_sep=multiple_val_sep,
    )
    turn_action_params = defaultdict(list)
    for service, formatted_actions in turn_actions_by_service.items():
        action_params = get_params(
            formatted_actions,
            include_values=include_values,
            slot_value_sep=slot_value_sep,
        )
        turn_action_params[service] = action_params
    return turn_action_params


def get_dialogue_outline(dialogue: dict) -> dict[str, list[dict[str, list[str]]]]:
    """
    Retrieves the dialogue outline, consisting of USER and SYSTEM acts, which are optionally parameterized by slots
    or slots and values.

    Parameters
    ----------
    dialogue
        Has the following structure::

            {
            'dialogue_id': str,
            'services': [str, ...], services (or APIs) that comprise the dialogue,
            'turns': [dict[Literal['frames', 'speaker', 'utterance'], Any], ...], turns with annotations. See
            `get_turn_actions` function for the structure of each element of the list.
            }

    Returns
    -------
    outline
        For each turn, a list comprising of the dialogue acts (e.g., INFORM, REQUEST) in that turn along with their
        parameters (e.g., 'food'='mexican', 'address').
    """
    outline = {"dialogue": [], "nlu": []}
    for i, turn in enumerate(dialogue["turns"], start=1):
        actions = get_turn_actions(turn)
        nlu_actions = get_turn_actions(turn.get("nlu", {}))
        outline["dialogue"].append(actions)
        outline["nlu"].append(nlu_actions)
    return outline


def get_intent_by_turn(dialogue: dict, exclude_none: bool = True) -> list[list[str]]:
    """Return the active intents in every dialogue turn. List returned has length
    of the number of user turns.

    Parameters
    ----------
    dialogue
        Nested dictionary of SGD-format dialogue.
    exclude_none
        If True, the `NONE` intent is not included in the intents set.


    Returns
    -------
    intents
        Each sublist contains the intents expressed by the user at a given turn.
    """
    intents = []
    for turn in dialogue_iterator(dialogue, user=True, system=False):
        this_turn_intents = []
        for frame in turn["frames"]:
            intent = frame["state"]["active_intent"]
            if exclude_none:
                if intent == "NONE":
                    continue
                else:
                    this_turn_intents.append(intent)
            else:
                this_turn_intents.append(intent)
        intents.append(this_turn_intents)

    return intents

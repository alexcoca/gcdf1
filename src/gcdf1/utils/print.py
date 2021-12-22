from __future__ import annotations

import json
import os.path

import numpy as np

from gcdf1.utils.dialogue import (
    get_dialogue_outline,
    get_intent_by_turn,
    get_utterances,
)

np.random.seed(0)


def print_dialogue(
    dialogue: dict, print_index: bool = False, show_intent: bool = False
):
    """
    Parameters
    ----------
    dialogue
        See `get_dialogue_outline` for structure.
    print_index
        If True, each turn will have its number printed.
    show_intent
        In each user turn, the active intent is appended.
    """

    utterances = get_utterances(dialogue)
    intents = []
    if show_intent:
        intents = get_intent_by_turn(dialogue)

    for i, utterance in enumerate(utterances):
        if show_intent and i % 2 == 0:
            utterance = f"{utterance} <intent> {' AND '.join(intents[i // 2])}"
        if print_index:
            print(f"{i + 1}: {utterance}")
        else:
            print(f"{utterance}")


def print_turn_outline(outline: dict[str, list[str]]):
    """
    Parameters
    ----------
    outline
        Output of `get_turn_actions`.
    """

    for service in outline:
        print(*outline[service], sep="\n")
        print("")


def print_dialogue_outline(
    dialogue: dict, text: bool = False, show_intent: bool = False
):
    """
    Parameters
    ----------
    dialogue
        See `get_dialogue_outline` for structure.
    text
        If `True`, also print the utterances alongside their outlines.
    show_intent
        If `True`, the intent is shown for each user utterance
    """
    outlines = get_dialogue_outline(dialogue)
    utterances = get_utterances(dialogue) if text else [""] * len(outlines["dialogue"])
    intents = get_intent_by_turn(dialogue) if show_intent else []
    has_nlu = any(action_dict for action_dict in outlines["nlu"])
    assert len(outlines["dialogue"]) == len(utterances)
    for i, (dial_outline, nlu_outline, utterance) in enumerate(
        zip(outlines["dialogue"], outlines["nlu"], utterances)
    ):
        if show_intent:
            utterance = f"{utterance} <intent> {' AND '.join(intents[i // 2])}"
        print(f"Turn: {i}:{utterance}")
        print_turn_outline(dial_outline)
        if has_nlu:
            print("#" * 15, " NLU ", "#" * 15)
            print_turn_outline(nlu_outline)
            print("")


if __name__ == "__main__":

    file = os.path.join("../../", "data/raw/train/dialogues_001.json")

    with open(file, "r") as f:
        all_dialogues = json.load(f)

    # print a random dialogue outline and its turns
    # NB: This does not work correctly for multiple frames in the same turn
    dialogue = all_dialogues[np.random.randint(0, high=len(all_dialogues))]
    print_dialogue(dialogue)
    print("")
    print_dialogue_outline(dialogue, text=True)
    print("")

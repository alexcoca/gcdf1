DOMAIN_CONSTRAINT_GOAL_KEYS = ["book", "fail_book", "info", "fail_info"]
"""Keys under which the constraints are stored in the goal for each domain. All keys are optional apart from `info`,
which should be left empty for dialogues which contain info requests. These model situations where there are no DB
results or bookings cannot be completed."""
DOMAIN_REQUEST_GOAL_KEY = "reqt"
"""Optional key under each domain if the user requests information about entities."""
MISSING_VALUE_MARKER = "?"
REF_SLOT_PATTERNS = ["ref", "Ref", "reference"]
DEPARTURE_TIME_SLOTS = ["leaveAt", "leaveat"]
ARRIVAL_TIME_SLOTS = ["arriveBy", "arriveby"]
GENERAL_ACT_PATTERNS = ["general"]
INFORM_ACT_PATTERNS = ["inform", "Inform", "INFORM"]
"""Patterns used together with `re.search` to return only INFORM dialogue acts from a frame.
"""
REQUEST_ACT_PATTERNS = ["request", "Request", "REQUEST"]
"""Patterns used together with `re.search` to return only REQUEST dialogue acts from a frame.
"""
TRANSACTION_FAILURE_ACT_PATTERNS = ["NoBook", "nobook"]
BOOKING_REQUEST_PATTERNS = ["booking-inform", "Booking-Inform"]
RECOMMENDATION_PATTERNS = ["Select", "select", "recommend", "Recommend"]
TRAIN_CONFIRMATION_PATTERNS = ["OfferBooked", "offerbooked"]
"""Patterns used together with `re.search` to return only OfferBooked dialogue acts from a frame."""
TRAIN_BOOKING_INTENT_PATTERNS = ["OfferBook$", "offerbook$"]
BOOK_ACT_PATTERNS = [r"\bbook$|\bBook$"]  # type: list
NOTIFY_FAILURE_PATTERNS = ["noofer", "NoOffer"]

SLOT_VALUE_ACT_PATTERNS = (
    INFORM_ACT_PATTERNS
    + RECOMMENDATION_PATTERNS
    + TRAIN_CONFIRMATION_PATTERNS
    + BOOK_ACT_PATTERNS
    + NOTIFY_FAILURE_PATTERNS
)

TRANSACTIONAL_DOMAINS = [
    "hotel",
    "Hotel",
    "restaurant",
    "Restaurant",
    "Train",
    "train",
]
"""Domains for which booking failures can occur."""
ENTITY_SLOTS = ["name", "Name"]
TRAIN_ENTITY_SLOTS = ["trainID", "trainid"]
ENTITY_SLOTS_BY_SERVICE = {
    "attraction": ENTITY_SLOTS,
    "restaurant": ENTITY_SLOTS,
    "hotel": ENTITY_SLOTS,
    "train": TRAIN_ENTITY_SLOTS,
    "taxi": [],
    "police": [],
    "hospital": [],
}
BOOKING_DOMAIN_PATTERNS = ["Booking", "booking"]
CHOICE_PATTERNS = ["choice", "Choice"]
"""Slots the system informs to present available options to the user."""

TAXI_CONFUSED_SLOTS = ["time", "type", "area", "pricerange", "name"]
MULTIPLE_VALUE_PROCESSING_EXCEPTIONS = ["address", "postcode", "phone"]
COREFERENCE_VALUES = {"restaurant": ["the restaurant"]}

_INFORMABLE_SLOTS = [
    "area",
    "arriveby",
    "day",
    "department",
    "departure",
    "destination",
    "food",
    "internet",
    "leaveat",
    "name",
    "parking",
    "people",
    "pricerange",
    "stars",
    "stay",
    "time",
    "type",
]

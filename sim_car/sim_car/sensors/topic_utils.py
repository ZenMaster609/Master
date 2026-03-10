"""Topic utilities for sim_car."""


def apply_topic_prefix(topic: str, prefix: str) -> str:
    """Apply a /sim or /sim/raw prefix to a topic path.

    If topic already starts with /sim/ or /sim/raw/, replace that prefix.
    Otherwise, treat topic as an absolute or relative path and prepend prefix.
    """
    if topic is None:
        return topic
    prefix = (prefix or '').rstrip('/')
    if not prefix:
        return topic

    if topic.startswith('/sim/raw/'):
        suffix = topic[len('/sim/raw'):]
    elif topic.startswith('/sim/'):
        suffix = topic[len('/sim'):]
    elif topic.startswith('/'):
        suffix = topic
    else:
        suffix = '/' + topic

    return prefix + suffix

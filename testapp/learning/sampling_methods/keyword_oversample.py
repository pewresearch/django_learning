from __future__ import absolute_import


def get_method():

    return {
        "sampling_strategy": "random",
        "stratify_by": None,
        "sampling_searches": {"test": {"pattern": "action", "proportion": 0.5}},
    }

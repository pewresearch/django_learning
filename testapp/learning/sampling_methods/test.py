

def get_method():

    return {
        "sampling_strategy": "random",
        "stratify_by": None,
        "sampling_searches": {
            'test': {
                "pattern": "test",
                "proportion": .5
            }
        }
    }
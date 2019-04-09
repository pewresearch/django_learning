def scorer(qual_assignment):

    return int(qual_assignment.codes.get(label__question__name="q1").label.value) == 1

